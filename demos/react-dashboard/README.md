# React Sentiment Dashboard (Workshop Demo)

## Quickstart

1. Install dependencies:
```bash
cd demos/react-dashboard
npm install
```

2.Run the dev server:
```bash
npm run dev
```

Default URL: `http://localhost:5173`

Ensure your Sentiment FastAPI is running (default http://localhost:8000/analyze). You can edit the API URL in the input at top-right.

**CSV upload**

Upload a CSV file with a text column. The dashboard will parse and call the API for each row.

**Notes**

This dashboard is intentionally minimal for workshop use.

For production, add robust error handling, authentication, rate-limiting and better visualizations (Chart.js, Recharts, etc.).


