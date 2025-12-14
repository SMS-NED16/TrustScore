# TrustScore Web UI

A simple web interface for the TrustScore pipeline, perfect for demos and quick evaluations.

## Quick Start

### 1. Install Dependencies

First, make sure you have the main TrustScore dependencies installed (from the root directory):
```bash
pip install -r requirements.txt
```

Then install the UI-specific dependencies:
```bash
cd ui
pip install -r requirements.txt
```

### 2. Run the Server

From the `ui` directory:
```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

## Usage

### Mock Mode (No API Calls)

For testing without making actual API calls:
1. Check the "Use Mock Mode" checkbox
2. Enter a prompt and response
3. Click "Analyze Response"

This will use mock data and is perfect for demos.

### Real Analysis

For real analysis:
1. Make sure you have your OpenAI API key set (optional, can be passed in request)
2. Leave "Use Mock Mode" unchecked
3. Enter a prompt and response
4. Click "Analyze Response"

## Features

- **Single Response Analysis**: Analyze individual prompt/response pairs
- **Visual Error Highlighting**: Errors are highlighted in the response text
- **Category Breakdown**: See Trustworthiness, Explainability, and Bias scores separately
- **Confidence Intervals**: View statistical uncertainty for all scores
- **Error Details**: Expandable error cards with explanations and severity

## API Endpoints

The backend also exposes REST API endpoints:

### POST `/api/analyze`
Analyze a single response:
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "model": "GPT-4o",
  "use_mock": false,
  "api_key": "optional-openai-key"
}
```

### POST `/api/analyze/batch`
Analyze multiple responses:
```json
{
  "samples": [
    {
      "prompt": "Question 1",
      "response": "Answer 1",
      "model": "GPT-4o"
    },
    {
      "prompt": "Question 2",
      "response": "Answer 2",
      "model": "GPT-4o"
    }
  ],
  "use_mock": false
}
```

### GET `/api/health`
Health check endpoint.

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, you can change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Change 5000 to your preferred port
```

### CORS Issues
CORS is enabled by default. If you encounter issues, check the `flask-cors` installation.

### Mock Mode Not Working
Make sure the TrustScore pipeline's mock components are properly configured in your `config/settings.py`.

## Demo Tips

1. **Use Mock Mode**: For a reliable demo, use mock mode to avoid API rate limits
2. **Prepare Examples**: Have a few example prompt/response pairs ready
3. **Show Error Highlighting**: Use examples with clear errors to demonstrate the highlighting feature
4. **Explain Scores**: Be ready to explain what TrustScore values mean (lower is better for errors)

## File Structure

```
ui/
├── app.py              # Flask backend server
├── static/
│   ├── index.html      # Main UI page
│   ├── styles.css      # Styling
│   └── app.js          # Frontend JavaScript
├── requirements.txt    # UI-specific dependencies
└── README.md           # This file
```








