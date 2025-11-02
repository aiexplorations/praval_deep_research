#!/usr/bin/env python3
"""
Paper Discovery Workflow Test Script

This script tests the paper discovery workflow end-to-end,
following CLAUDE.md guidelines for production-ready testing.

Usage:
    python scripts/test_paper_discovery.py [--integration]
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from agents.research.paper_discovery import paper_discovery_agent
    from processors.arxiv_client import ArXivClient, search_arxiv_papers
    from praval import Spore, broadcast
    PRAVAL_AVAILABLE = True
    print("✅ All dependencies loaded successfully")
except ImportError as e:
    print(f"⚠️  Praval/dependencies not available: {e}")
    PRAVAL_AVAILABLE = False


class PaperDiscoveryTester:
    """Test harness for paper discovery workflow."""
    
    def __init__(self):
        self.test_results = []
        self.broadcasts = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅" if success else "❌"
        self.test_results.append({
            "name": test_name,
            "success": success,
            "details": details
        })
        print(f"{status} {test_name}: {details}")
    
    def mock_broadcast(self, message: Dict[str, Any]):
        """Mock broadcast function to capture agent communications."""
        self.broadcasts.append(message)
        print(f"📡 Broadcast: {message.get('type', 'unknown')}")
    
    async def test_arxiv_client_basic(self) -> bool:
        """Test basic ArXiv client functionality."""
        try:
            client = ArXivClient()
            
            # Test basic configuration
            assert client.base_url == "http://export.arxiv.org/api/query"
            assert client.max_results > 0
            
            # Test query building
            query = client._build_search_query("machine learning")
            assert "machine+learning" in query
            
            self.log_test("ArXiv Client Basic", True, "Configuration and query building OK")
            return True
            
        except Exception as e:
            self.log_test("ArXiv Client Basic", False, f"Error: {e}")
            return False
    
    async def test_arxiv_search_tool(self) -> bool:
        """Test ArXiv search tool function."""
        try:
            # Test the high-level search function
            papers = await search_arxiv_papers(
                query="neural networks",
                max_results=2,
                domain="computer_science"
            )
            
            # Validate results
            assert isinstance(papers, list)
            
            if papers:
                paper = papers[0]
                required_fields = ['title', 'authors', 'abstract', 'arxiv_id', 'url']
                for field in required_fields:
                    assert field in paper, f"Missing field: {field}"
                
                self.log_test("ArXiv Search Tool", True, 
                            f"Found {len(papers)} papers with valid structure")
            else:
                self.log_test("ArXiv Search Tool", True, 
                            "No papers found but search completed successfully")
            
            return True
            
        except Exception as e:
            self.log_test("ArXiv Search Tool", False, f"Error: {e}")
            return False
    
    def test_agent_structure(self) -> bool:
        """Test paper discovery agent structure."""
        if not PRAVAL_AVAILABLE:
            self.log_test("Agent Structure", False, "Praval not available")
            return False
        
        try:
            # Check agent function exists and is callable
            assert callable(paper_discovery_agent)
            
            # Check docstring has identity statement
            docstring = paper_discovery_agent.__doc__ or ""
            assert "I am" in docstring, "Agent should have identity statement"
            
            self.log_test("Agent Structure", True, "Agent properly structured with identity")
            return True
            
        except Exception as e:
            self.log_test("Agent Structure", False, f"Error: {e}")
            return False
    
    async def test_agent_spore_processing(self) -> bool:
        """Test agent processes spores correctly."""
        if not PRAVAL_AVAILABLE:
            self.log_test("Agent Spore Processing", False, "Praval not available")
            return False
        
        try:
            # Create test spore
            test_spore = Spore()
            test_spore.knowledge = {
                "type": "search_request",
                "query": "machine learning",
                "domain": "computer_science",
                "max_results": 2,
                "session_id": "test_session"
            }
            
            # Mock dependencies and capture broadcasts
            import agents.research.paper_discovery as agent_module
            
            original_broadcast = getattr(agent_module, 'broadcast', None)
            original_chat = getattr(agent_module, 'chat', None)
            
            # Replace with mocks
            agent_module.broadcast = self.mock_broadcast
            agent_module.chat = lambda x: "optimized query: machine learning algorithms"
            
            try:
                # Execute agent
                result = paper_discovery_agent(test_spore)
                
                # Check if broadcasts were made
                if self.broadcasts:
                    broadcast = self.broadcasts[-1]
                    assert broadcast.get("type") == "papers_found"
                    assert "knowledge" in broadcast
                    
                    self.log_test("Agent Spore Processing", True, 
                                "Agent processed spore and broadcasted results")
                else:
                    self.log_test("Agent Spore Processing", True, 
                                "Agent processed spore (no broadcasts in test mode)")
                
                return True
                
            finally:
                # Restore original functions
                if original_broadcast:
                    agent_module.broadcast = original_broadcast
                if original_chat:
                    agent_module.chat = original_chat
                    
        except Exception as e:
            self.log_test("Agent Spore Processing", False, f"Error: {e}")
            return False
    
    def test_agent_memory(self) -> bool:
        """Test agent memory functionality."""
        if not PRAVAL_AVAILABLE:
            self.log_test("Agent Memory", False, "Praval not available")
            return False
        
        try:
            # Check if agent has memory methods
            if hasattr(paper_discovery_agent, 'remember') and \
               hasattr(paper_discovery_agent, 'recall'):
                
                # Test memory operations
                test_memory = "test_discovery: machine learning -> 3 papers found"
                
                try:
                    paper_discovery_agent.remember(test_memory, importance=0.8)
                    memories = paper_discovery_agent.recall("test_discovery", limit=1)
                    
                    self.log_test("Agent Memory", True, 
                                f"Memory operations successful, found {len(memories) if memories else 0} memories")
                    return True
                    
                except Exception as mem_error:
                    self.log_test("Agent Memory", False, 
                                f"Memory operations failed: {mem_error}")
                    return False
            else:
                self.log_test("Agent Memory", False, "Agent missing memory methods")
                return False
                
        except Exception as e:
            self.log_test("Agent Memory", False, f"Error: {e}")
            return False
    
    async def test_integration_workflow(self) -> bool:
        """Test complete integration workflow with real API."""
        if not PRAVAL_AVAILABLE:
            self.log_test("Integration Workflow", False, "Praval not available")
            return False
        
        try:
            print("\n🔄 Running integration test with real ArXiv API...")
            
            # Create realistic spore
            search_spore = Spore()
            search_spore.knowledge = {
                "type": "search_request",
                "query": "transformer neural networks",
                "domain": "artificial_intelligence",
                "max_results": 3,
                "session_id": "integration_test",
                "quality_threshold": 0.3
            }
            
            # Mock broadcast to capture results
            import agents.research.paper_discovery as agent_module
            original_broadcast = getattr(agent_module, 'broadcast', None)
            agent_module.broadcast = self.mock_broadcast
            
            try:
                # Execute agent with real API
                start_time = time.time()
                result = paper_discovery_agent(search_spore)
                end_time = time.time()
                
                # Validate results
                if self.broadcasts:
                    broadcast = self.broadcasts[-1]
                    
                    if broadcast.get("type") == "papers_found":
                        knowledge = broadcast["knowledge"]
                        papers = knowledge.get("papers", [])
                        
                        self.log_test("Integration Workflow", True,
                                    f"Found {len(papers)} papers in {end_time-start_time:.2f}s")
                        
                        # Show sample results
                        if papers:
                            print(f"📄 Sample paper: {papers[0].get('title', 'Unknown')[:60]}...")
                        
                        return True
                    else:
                        self.log_test("Integration Workflow", False,
                                    f"Unexpected broadcast type: {broadcast.get('type')}")
                        return False
                else:
                    self.log_test("Integration Workflow", False, "No broadcasts received")
                    return False
                    
            finally:
                if original_broadcast:
                    agent_module.broadcast = original_broadcast
                    
        except Exception as e:
            self.log_test("Integration Workflow", False, f"Error: {e}")
            return False
    
    async def run_all_tests(self, include_integration: bool = False):
        """Run all tests."""
        print("🧪 Starting Paper Discovery Workflow Tests")
        print("=" * 50)
        
        # Basic tests
        await self.test_arxiv_client_basic()
        await self.test_arxiv_search_tool()
        self.test_agent_structure()
        await self.test_agent_spore_processing()
        self.test_agent_memory()
        
        # Integration test (optional)
        if include_integration:
            await self.test_integration_workflow()
        else:
            print("\n⏭️  Skipping integration test (use --integration to enable)")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Check details above.")
            return False


async def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test paper discovery workflow")
    parser.add_argument("--integration", action="store_true", 
                       help="Include integration tests with real ArXiv API")
    
    args = parser.parse_args()
    
    tester = PaperDiscoveryTester()
    success = await tester.run_all_tests(include_integration=args.integration)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())