import requests
import spacy
import base64
from config import GITHUB_TOKEN, GITHUB_USERNAME

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def fetch_github_data(username: str, token: str) -> dict:
    """Fetch GitHub user profile and repo READMEs."""
    headers = {"Authorization": f"token {token}"}
    github_data = {"username": username, "repos": []}

    # Fetch user profile
    profile_url = f"https://api.github.com/users/{username}"
    profile_resp = requests.get(profile_url, headers=headers)
    profile_resp.raise_for_status()
    github_data["profile"] = profile_resp.json().get("bio", "")

    # Fetch public repos
    repos_url = f"https://api.github.com/users/{username}/repos"
    repos_resp = requests.get(repos_url, headers=headers)
    repos_resp.raise_for_status()
    repos = repos_resp.json()

    # Fetch README for each repo (limit to 3 for simplicity)
    for repo in repos[:3]:
        repo_name = repo["name"]
        readme_url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
        readme_resp = requests.get(readme_url, headers=headers)
        if readme_resp.status_code == 200:
            readme_content = base64.b64decode(readme_resp.json()["content"]).decode("utf-8")
            github_data["repos"].append({"name": repo_name, "readme": readme_content})
        else:
            github_data["repos"].append({"name": repo_name, "readme": ""})

    return github_data

def preprocess_github_data(github_data: dict) -> dict:
    """Preprocess GitHub profile and repo READMEs with spaCy."""
    processed_data = {"username": github_data["username"], "repos": []}

    # Process profile bio
    profile_doc = nlp(github_data["profile"] or "No bio provided")
    processed_data["profile"] = {
        "tokens": [token.text.lower() for token in profile_doc if not token.is_punct and not token.is_stop],
        "entities": [(ent.text, ent.label_) for ent in profile_doc.ents]
    }

    # Process repo READMEs
    for repo in github_data["repos"]:
        readme_doc = nlp(repo["readme"] or "No README content")
        processed_data["repos"].append({
            "name": repo["name"],
            "tokens": [token.text.lower() for token in readme_doc if not token.is_punct and not token.is_stop],
            "entities": [(ent.text, ent.label_) for ent in readme_doc.ents]
        })

    return processed_data

def main():
    # Fetch and preprocess GitHub data
    github_data = fetch_github_data(GITHUB_USERNAME, GITHUB_TOKEN)
    processed_data = preprocess_github_data(github_data)

    # Print results
    print("GitHub Preprocessing:")
    print(f"Profile Tokens: {processed_data['profile']['tokens']}")
    print(f"Profile Entities: {processed_data['profile']['entities']}")
    for repo in processed_data["repos"]:
        print(f"\nRepo: {repo['name']}")
        print(f"Tokens: {repo['tokens']}")
        print(f"Entities: {repo['entities']}")
    print("\nGitHub data fetch and preprocessing successful!")

if __name__ == "__main__":
    main()
