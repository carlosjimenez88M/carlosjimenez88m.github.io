#!/bin/bash

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# Build the project with Hugo
hugo

# Copy generated files from public/ to root
echo -e "\033[0;34mCopying files from public/ to root...\033[0m"
cp -r public/* .

# Add all changes to git
git add -A

# Commit changes
msg="rebuilding site `date`"
if [ $# -eq 1 ]
  then msg="$1"
fi
git commit -m "$msg"

# Push to GitHub
git push origin master

echo -e "\033[0;32mDone! Your changes will be live in 2-5 minutes.\033[0m"
