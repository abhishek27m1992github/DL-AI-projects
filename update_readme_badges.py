import os
import re

# Replace these with your actual GitHub username/org and repo name
GITHUB_OWNER = "YOUR_GITHUB_USERNAME_OR_ORG"
GITHUB_REPO = "YOUR_REPO_NAME"

WORKFLOWS_DIR = os.path.join(".github", "workflows")
PROJECTS_DIR = "."

def get_workflow_files():
    if not os.path.isdir(WORKFLOWS_DIR):
        return []
    return [
        f for f in os.listdir(WORKFLOWS_DIR)
        if f.endswith(".yml") or f.endswith(".yaml")
    ]

def get_badge_line(workflow_file):
    return f"![Build Status](https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{workflow_file}/badge.svg)\n"

def update_readme(project_dir, workflow_file):
    readme_path = os.path.join(project_dir, "README.md")
    badge_line = get_badge_line(workflow_file)
    if not os.path.exists(readme_path):
        return
    with open(readme_path, "r") as f:
        lines = f.readlines()
    # Remove any existing badge line
    lines = [line for line in lines if not re.match(r"!\\[Build Status\\]\\(https://github\\.com/.*/actions/workflows/.*badge\\.svg\\)", line)]
    # Insert badge at the top
    lines = [badge_line] + lines
    with open(readme_path, "w") as f:
        f.writelines(lines)
    print(f"Updated badge in {readme_path}")

def main():
    workflow_files = get_workflow_files()
    for workflow_file in workflow_files:
        # Assume workflow file is named like 'project1-ci.yml' or 'project2-ci.yml'
        project_name = workflow_file.split("-")[0]
        project_dir = os.path.join(PROJECTS_DIR, project_name)
        if os.path.isdir(project_dir):
            update_readme(project_dir, workflow_file)

if __name__ == "__main__":
    main() 