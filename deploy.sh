#!/bin/bash

set -e  # Exit on error

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# Check if we're in the right directory
if [ ! -f "hugo.toml" ]; then
    echo -e "\033[0;31mError: hugo.toml not found. Are you in the right directory?\033[0m"
    exit 1
fi

# Build the project with Hugo
echo -e "\033[0;34mBuilding site with Hugo...\033[0m"
hugo --cleanDestinationDir --minify

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "\033[0;31mError: Hugo build failed!\033[0m"
    exit 1
fi

# Verify that public directory exists and has content
if [ ! -d "public" ] || [ -z "$(ls -A public)" ]; then
    echo -e "\033[0;31mError: public directory is empty or doesn't exist!\033[0m"
    exit 1
fi

# Copy generated files from public/ to root
echo -e "\033[0;34mCopying files from public/ to root...\033[0m"
cp -r public/* .

# Verify critical files were copied
for file in index.html algolia.json index.xml; do
    if [ ! -f "$file" ]; then
        echo -e "\033[0;33mWarning: $file not found in root!\033[0m"
    fi
done

# Add all changes to git
echo -e "\033[0;34mStaging changes...\033[0m"
git add -A

# Commit changes
msg="rebuilding site `date`"
if [ $# -eq 1 ]; then
    msg="$1"
fi

echo -e "\033[0;34mCommitting with message: $msg\033[0m"
git commit -m "$msg" || echo -e "\033[0;33mNothing to commit (no changes)\033[0m"

# Push to GitHub
echo -e "\033[0;34mPushing to GitHub...\033[0m"
git push origin master

echo -e "\033[0;32m✓ Done! Your changes will be live in 2-5 minutes.\033[0m"
echo -e "\033[0;32m✓ Site: https://carlosdanieljimenez.com/\033[0m"
