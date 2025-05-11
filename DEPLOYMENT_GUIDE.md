# Deployment Guide for Email Spam Shield

This guide will help you deploy your Email Spam Shield project to GitHub and optionally to Streamlit Sharing for public access.

## GitHub Deployment

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com/) and sign in to your account.
2. Click on the "+" icon in the top-right corner and select "New repository".
3. Enter "email-spam-shield" as the repository name.
4. Add a short description: "A machine learning application for spam email detection."
5. Keep the repository as "Public" (or choose "Private" if you prefer).
6. Click "Create repository".

### Step 2: Push Your Code to GitHub

After creating the repository, GitHub will show you the commands to push your code. Run these commands in your terminal:

```bash
# Replace 'your-username' with your actual GitHub username
git remote add origin https://github.com/your-username/email-spam-shield.git
git branch -M main
git push -u origin main
```

### Step 3: Verify Your Repository

1. Refresh your GitHub repository page to see your files.
2. Check that all essential files (app.py, models, etc.) are present.
3. The README.md file should be rendered on the main page of your repository.

## Streamlit Cloud Deployment (Optional)

If you want to make your application accessible online for anyone to use, you can deploy it to Streamlit Cloud:

### Step 1: Create a Streamlit Cloud Account

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign up for an account.
2. Link your GitHub account during registration.

### Step 2: Deploy Your App

1. From your Streamlit Cloud dashboard, click "New app".
2. Select your GitHub repository (email-spam-shield).
3. Set the main file path to "app.py".
4. Choose the Python version (3.9+).
5. Click "Deploy".

### Step 3: Configure Advanced Settings (If Needed)

If your app has specific requirements:

1. From your app's menu, select "Settings".
2. You can adjust:
   - Memory/CPU requirements
   - Package caching
   - Secrets management (for API keys, etc.)

### Notes on Model Files

The model files (*.pkl) in your repository are quite large. Streamlit Cloud has some limitations:

1. GitHub has file size limitations (100MB per file, 2GB per repository).
2. If your model files exceed these limits, consider:
   - Using Git LFS (Large File Storage)
   - Storing models in cloud storage (AWS S3, Google Cloud Storage) and downloading at runtime
   - Using a model compression technique

## Updating Your Deployment

Whenever you make changes to your code:

1. Commit the changes locally:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

2. Push to GitHub:
   ```bash
   git push
   ```

3. If deployed on Streamlit Cloud, it will automatically rebuild your app with the new changes.

## Troubleshooting

- **Package installation issues**: Check if all dependencies are correctly specified in requirements.txt.
- **Model loading errors**: Ensure paths to model files are correct and files are properly committed.
- **Memory errors**: Large models might exceed memory limits on free tiers of hosting platforms.
- **Authentication issues**: You may need to use GitHub PAT (Personal Access Token) for authentication. 