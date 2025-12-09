-- Migration: Thread-based branching
-- Replaces the tree-based branching model with a simpler thread-based model
--
-- In this model:
-- - Each message has a thread_id (integer, 0 = original conversation)
-- - Each message has a position (1-indexed position within the thread)
-- - Editing creates a new thread that copies messages up to the edit point
-- - Each thread is a complete, independent conversation path

-- Step 1: Add new columns to conversations table
ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS active_thread_id INTEGER DEFAULT 0 NOT NULL;

ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS max_thread_id INTEGER DEFAULT 0 NOT NULL;

-- Step 2: Add new columns to messages table
ALTER TABLE messages
ADD COLUMN IF NOT EXISTS thread_id INTEGER DEFAULT 0 NOT NULL;

ALTER TABLE messages
ADD COLUMN IF NOT EXISTS position INTEGER;

-- Step 3: Migrate existing messages - assign positions based on timestamp
-- For each conversation, number messages sequentially
WITH numbered_messages AS (
    SELECT
        id,
        conversation_id,
        ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY timestamp) as new_position
    FROM messages
    WHERE position IS NULL
)
UPDATE messages m
SET position = nm.new_position
FROM numbered_messages nm
WHERE m.id = nm.id;

-- Step 4: Make position NOT NULL after migration
ALTER TABLE messages
ALTER COLUMN position SET NOT NULL;

-- Step 5: Drop old branching columns (if they exist)
ALTER TABLE messages DROP COLUMN IF EXISTS parent_message_id;
ALTER TABLE messages DROP COLUMN IF EXISTS branch_id;
ALTER TABLE messages DROP COLUMN IF EXISTS branch_index;
ALTER TABLE conversations DROP COLUMN IF EXISTS active_branch_id;

-- Step 6: Drop old indexes
DROP INDEX IF EXISTS ix_messages_conversation_branch;
DROP INDEX IF EXISTS ix_messages_parent;

-- Step 7: Create new indexes for thread-based queries
CREATE INDEX IF NOT EXISTS ix_messages_conversation_thread
ON messages(conversation_id, thread_id, position);

CREATE INDEX IF NOT EXISTS ix_messages_conversation_position
ON messages(conversation_id, position);

-- Step 8: Add constraints
ALTER TABLE conversations
ADD CONSTRAINT conversations_active_thread_non_negative
CHECK (active_thread_id >= 0);

ALTER TABLE conversations
ADD CONSTRAINT conversations_max_thread_non_negative
CHECK (max_thread_id >= 0);

ALTER TABLE messages
ADD CONSTRAINT messages_thread_non_negative
CHECK (thread_id >= 0);

ALTER TABLE messages
ADD CONSTRAINT messages_position_positive
CHECK (position > 0);
