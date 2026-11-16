# OncoLens EDU — ready to deploy to Render

This directory contains only the runnable research demonstration, not the report or presentation. The model, demo cases and required benchmark dataset are included. There is no need to train the model on Render.

**Not for clinical use.** Do not submit actual patient information. This is an historical educational benchmark, not a medical diagnostic tool.

## Deploy through GitHub + Render
1. Create a new GitHub repository (private if preferred).
2. Upload the **contents of this directory** to the repository root, preserving `models/` and `data/` folders. Do not upload the enclosing directory as an extra level.
3. Visit https://dashboard.render.com/ and create **New > Web Service**.
4. Connect the GitHub repository and select it.
5. Choose **Python 3** and **Free** instance, with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Health check path (optional): `/health`
   - Python version: `3.12.8` (in `render.yaml`; if configuring manually, set `PYTHON_VERSION` accordingly).
6. Deploy and open the URL Render gives you.
7. Visit `/health` and `/docs`; test a demo sample and click **Evaluate complete feature vector**.
8. Capture your own live deployment screenshots and update the report's deployment section and presentation URL. Do not claim online deployment before it works.

**Important:** The free Render service may sleep after 15 minutes of inactivity, so open it before presenting. Public deployment also makes the name and student ID visible on the homepage. Review those before you publish.
