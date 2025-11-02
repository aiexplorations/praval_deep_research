#!/bin/bash
cd /Users/rajesh/Github/praval_deep_research

# Stage the changes
git add -A

# Commit the rename
git commit -m "Rename project from agentic_deep_research to praval_deep_research

- Updated app name in config.py
- Updated Docker environment variables
- Updated test assertions
- Folder renamed to match Praval branding"

# Create GitHub repo and push
gh repo create praval_deep_research \
  --private \
  --source=. \
  --description "Local-First AI Research Assistant for ArXiv Papers - Built with Praval Agentic Framework" \
  --push

echo "Repository created and pushed successfully!"
