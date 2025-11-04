/**
 * Knowledge Base Page - Manage Indexed Papers
 *
 * View, search, and manage all indexed papers in the knowledge base.
 */

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
  type ColumnFiltersState
} from '@tanstack/react-table';
import { apiClient } from '../services/api/client';
import type { Paper } from '../types';

const columnHelper = createColumnHelper<Paper>();

export default function KnowledgeBase() {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [globalFilter, setGlobalFilter] = useState('');

  const queryClient = useQueryClient();

  // Fetch papers
  const { data: papersData, isLoading, error } = useQuery({
    queryKey: ['knowledge-base-papers'],
    queryFn: () => apiClient.listPapers()
  });

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ['knowledge-base-stats'],
    queryFn: () => apiClient.getKnowledgeBaseStats()
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (paperId: string) => apiClient.deletePaper(paperId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] });
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-stats'] });
    }
  });

  // Clear mutation
  const clearMutation = useMutation({
    mutationFn: () => apiClient.clearKnowledgeBase(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] });
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-stats'] });
    }
  });

  const handleDelete = (paperId: string, title: string) => {
    if (confirm(`Delete "${title}"?`)) {
      deleteMutation.mutate(paperId);
    }
  };

  const handleClearAll = () => {
    if (confirm('⚠️ This will delete ALL papers and vectors. Are you sure?')) {
      if (confirm('This action cannot be undone. Proceed?')) {
        clearMutation.mutate();
      }
    }
  };

  const columns = [
    columnHelper.accessor('title', {
      header: 'Title',
      cell: (info) => (
        <div className="max-w-md">
          <p className="font-medium text-sm line-clamp-2">{info.getValue()}</p>
        </div>
      )
    }),
    columnHelper.accessor('authors', {
      header: 'Authors',
      cell: (info) => {
        const authors = info.getValue();
        return (
          <div className="text-sm text-muted-foreground">
            {authors.slice(0, 2).join(', ')}
            {authors.length > 2 && ` +${authors.length - 2}`}
          </div>
        );
      }
    }),
    columnHelper.accessor('chunk_count', {
      header: 'Chunks',
      cell: (info) => (
        <div className="text-sm text-center">{info.getValue() || 'N/A'}</div>
      )
    }),
    columnHelper.accessor('categories', {
      header: 'Category',
      cell: (info) => {
        const categories = info.getValue();
        return (
          <div className="text-xs text-muted-foreground">
            {categories && categories[0] ? categories[0] : 'N/A'}
          </div>
        );
      }
    }),
    columnHelper.display({
      id: 'actions',
      header: 'Actions',
      cell: (info) => (
        <button
          onClick={() => handleDelete(info.row.original.id, info.row.original.title)}
          disabled={deleteMutation.isPending}
          className="text-xs px-2 py-1 text-destructive hover:bg-destructive/10 rounded transition-colors disabled:opacity-50"
        >
          Delete
        </button>
      )
    })
  ];

  const table = useReactTable({
    data: papersData?.papers || [],
    columns,
    state: {
      sorting,
      columnFilters,
      globalFilter
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel()
  });

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Knowledge Base</h1>
          <p className="text-muted-foreground">
            Manage your indexed research papers
          </p>
        </div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Total Papers</p>
              <p className="text-2xl font-bold">{stats.total_papers}</p>
            </div>
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Total Vectors</p>
              <p className="text-2xl font-bold">{stats.total_vectors.toLocaleString()}</p>
            </div>
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Avg Chunks/Paper</p>
              <p className="text-2xl font-bold">{stats.avg_chunks_per_paper.toFixed(1)}</p>
            </div>
            <div className="p-4 border border-border rounded-lg bg-card">
              <p className="text-sm text-muted-foreground mb-1">Categories</p>
              <p className="text-2xl font-bold">{Object.keys(stats.categories).length}</p>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="mb-6 flex flex-col md:flex-row gap-4 items-center justify-between">
          <input
            type="text"
            value={globalFilter ?? ''}
            onChange={(e) => setGlobalFilter(e.target.value)}
            placeholder="Search papers..."
            className="w-full md:w-96 px-4 py-2 border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-ring"
          />
          <div className="flex gap-2">
            <button
              onClick={() => queryClient.invalidateQueries({ queryKey: ['knowledge-base-papers'] })}
              className="px-4 py-2 border border-border rounded-lg hover:bg-muted transition-colors"
            >
              🔄 Refresh
            </button>
            <button
              onClick={handleClearAll}
              disabled={clearMutation.isPending || !papersData?.papers.length}
              className="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Clear All Papers
            </button>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive rounded-lg mb-6">
            <p className="text-destructive font-medium">Error loading papers</p>
            <p className="text-sm text-destructive/80">{(error as any)?.message}</p>
          </div>
        )}

        {/* Table */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-border border-t-primary"></div>
            <p className="mt-4 text-muted-foreground">Loading papers...</p>
          </div>
        ) : papersData?.papers.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg">No papers in knowledge base</p>
            <p className="text-sm mt-2">Search and index papers from the Discover page</p>
          </div>
        ) : (
          <div className="border border-border rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted">
                  {table.getHeaderGroups().map((headerGroup) => (
                    <tr key={headerGroup.id}>
                      {headerGroup.headers.map((header) => (
                        <th
                          key={header.id}
                          className="px-4 py-3 text-left text-sm font-medium text-muted-foreground cursor-pointer hover:bg-muted/80"
                          onClick={header.column.getToggleSortingHandler()}
                        >
                          <div className="flex items-center gap-2">
                            {flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                            {{
                              asc: '↑',
                              desc: '↓'
                            }[header.column.getIsSorted() as string] ?? null}
                          </div>
                        </th>
                      ))}
                    </tr>
                  ))}
                </thead>
                <tbody className="divide-y divide-border">
                  {table.getRowModel().rows.map((row) => (
                    <tr key={row.id} className="hover:bg-muted/50">
                      {row.getVisibleCells().map((cell) => (
                        <td key={cell.id} className="px-4 py-3">
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext()
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Results Count */}
        {papersData && papersData.papers.length > 0 && (
          <div className="mt-4 text-sm text-muted-foreground text-center">
            Showing {table.getRowModel().rows.length} of {papersData.papers.length} papers
          </div>
        )}
      </div>
    </div>
  );
}
