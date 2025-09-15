import json
import spacy
from pathlib import Path
from config import GITHUB_TOKEN, GITHUB_USERNAME
import requests
import base64
from utils.utils import clean_text

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
            readme_content = clean_text(base64.b64decode(readme_resp.json()["content"]).decode("utf-8"))
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

def generate_inputs():
    """Generate inputs.json from vacancy.txt, user_bio.txt, and GitHub data."""
    # Load vacancy and user bio
    vacancy_path = "data/vacancy.txt"
    user_bio_path = "data/user_bio.txt"

    try:
        vacancy_text = clean_text(Path(vacancy_path).read_text(encoding="utf-8"))
        user_bio_text = clean_text(Path(user_bio_path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e}")

    # Fetch GitHub data
    github_data = fetch_github_data(GITHUB_USERNAME, GITHUB_TOKEN)

    # Structure data for JSON
    inputs = {
        "vacancy": vacancy_text,
        "user_bio": user_bio_text,
        "github_profile": github_data["profile"],
        "github_repos": github_data["repos"]
    }

    # Save to inputs.json
    output_path = "data/inputs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inputs, f, indent=2)

    # Verify by reading back
    with open(output_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    print("Generated inputs.json:")
    print(f"Vacancy length: {len(saved_data['vacancy'])} characters")
    print(f"User bio length: {len(saved_data['user_bio'])} characters")
    print(f"GitHub profile: {saved_data['github_profile']}")
    print(f"GitHub repos: {[repo['name'] for repo in saved_data['github_repos']]}")
    print("\nInputs generation successful!")

if __name__ == "__main__":
    generate_inputs()
