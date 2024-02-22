"""Main script to run the dash dashboard."""

from dashboard.dashboard import app, create_layout

if __name__ == "__main__":
    app.layout = create_layout()
    app.run_server(debug=True)
