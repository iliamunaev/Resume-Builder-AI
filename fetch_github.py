import requests
import spacy
import base64
from utils.utils import clean_text
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

def spacy_tokens_and_ents(text: str) -> dict:
    """Clean -> nlp -> tokens/entities."""
    cleaned = clean_text(text or "")
    doc = nlp(cleaned if cleaned.strip() else "no content")
    tokens = [t.lemma_.lower() for t in doc if not t.is_punct and not t.is_stop]
    ents = [(e.text, e.label_) for e in doc.ents]
    return {"tokens": tokens, "entities": ents}

def preprocess_github_data(github_data: dict) -> dict:
    """Preprocess GitHub profile and repo READMEs with spaCy."""
    out = {"username": github_data["username"], "repos": []}
    out["profile"] = spacy_tokens_and_ents(github_data.get("profile", ""))

    for repo in github_data.get("repos", []):
        repo_out = spacy_tokens_and_ents(repo.get("readme", ""))
        repo_out["name"] = repo.get("name", "")
        out["repos"].append(repo_out)
    return out

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
