
# Simple Web Server Example

This is an example of a traditional repository that uses task prompts instead of ai-module.yaml.

## Task Prompt

"Create a simple web server that serves static files and provides a basic API for cryptocurrency prices. The server should be built with Python Flask and include basic logging and error handling."

## Repository Structure

```
simple-web-server/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/            # Static files
│   ├── index.html
│   └── style.css
├── templates/         # HTML templates
│   └── dashboard.html
└── README.md          # This file
```

## Expected Deployment

The orchestrator will:
1. Detect this as a task prompt workflow (no ai-module.yaml)
2. Analyze the repository structure
3. Generate deployment strategy based on detected technology stack
4. Install dependencies from requirements.txt
5. Start the Flask application
6. Setup basic monitoring

## Manual Deployment

If deploying manually:

```bash
pip install -r requirements.txt
python app.py
```

The server will start on port 5000.
