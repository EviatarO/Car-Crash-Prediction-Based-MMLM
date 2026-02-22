# Push this repo to GitHub

Your local repo is ready (initial commit on branch `main`). To create the remote and push:

## 1. Log in to GitHub (one-time)

In a terminal, run:

```bash
gh auth login
```

- Choose **GitHub.com**
- Choose **HTTPS**
- Choose **Login with a web browser** (or paste a token if you prefer)
- Complete the steps in the browser

## 2. Create the private repo and push

**Option A – GitHub CLI (after `gh auth login`):**

```bash
cd /home/eprojuser011/MMLM_CursorAI
gh repo create MMLM_CursorAI --private --source=. --remote=origin --push
```

**Option B – Create repo on the website, then push:**  
Create an empty private repo named **MMLM_CursorAI** at https://github.com/new (no README). The `origin` remote is already set. Then:

```bash
cd /home/eprojuser011/MMLM_CursorAI
git push -u origin main
```

## Alternative: create repo on the website, then push

1. Go to https://github.com/new
2. Repository name: **MMLM_CursorAI**
3. Set to **Private**, do not add a README
4. Then run:

```bash
cd /home/eprojuser011/MMLM_CursorAI
git remote add origin https://github.com/EviatarO/MMLM_CursorAI.git
git push -u origin main
```

If GitHub asks for credentials, use a **Personal Access Token** (Settings → Developer settings → Personal access tokens) as the password.
