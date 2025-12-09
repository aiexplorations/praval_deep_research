-- Migration: Add branching support to messages table
-- Run this migration to enable edit/resubmit conversation branching

-- Add active_branch_id to conversations table
ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS active_branch_id UUID DEFAULT NULL;

-- Add branching columns to messages table
ALTER TABLE messages
ADD COLUMN IF NOT EXISTS parent_message_id UUID DEFAULT NULL REFERENCES messages(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS branch_id UUID DEFAULT NULL,
ADD COLUMN IF NOT EXISTS branch_index INTEGER DEFAULT 0 NOT NULL;

-- Create indexes for efficient branch queries
CREATE INDEX IF NOT EXISTS ix_messages_conversation_branch
ON messages(conversation_id, branch_id);

CREATE INDEX IF NOT EXISTS ix_messages_parent
ON messages(parent_message_id);

-- Comment explaining the branching model
COMMENT ON COLUMN messages.parent_message_id IS 'Points to the message this is a reply/edit of (tree structure)';
COMMENT ON COLUMN messages.branch_id IS 'UUID grouping messages in the same branch (null = main branch)';
COMMENT ON COLUMN messages.branch_index IS 'Position among sibling branches (0 = original, 1+ = edits)';
COMMENT ON COLUMN conversations.active_branch_id IS 'Currently displayed branch (null = main branch)';
