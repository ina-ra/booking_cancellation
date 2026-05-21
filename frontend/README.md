# Frontend

This directory contains the Vite + React frontend for the booking cancellation service.

## Run locally

1. Install Node.js 20 or newer.
2. Install dependencies:

```powershell
cd frontend
npm install
```

3. Start the FastAPI backend from the repository root:

```powershell
py -m uvicorn src.interfaces.main:app --reload
```

4. In a second terminal, start the frontend:

```powershell
cd frontend
npm run dev
```

The Vite dev server runs on `http://127.0.0.1:5173` and proxies `/frontend-api/*` requests to FastAPI on `http://127.0.0.1:8000`.
