# main.py (Add imports and middleware)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware # <--- Import CORS middleware
import backend_logic
import os
import shutil

app = FastAPI()

# --- CORS Configuration ---
origins = [
    "http://localhost",         # Allow frontend served from localhost (any port)
    "http://localhost:8080",    # Example: If you use a separate frontend dev server
    "http://127.0.0.1",         # Allow frontend served from 127.0.0.1 (any port)
    "http://127.0.0.1:8000",    # Allow requests from the same origin
    "null"                      # Allow requests from 'file://' origin (opening HTML directly) - Use with caution!
    # Add any other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # List of allowed origins
    allow_credentials=True,        # Allow cookies (if needed)
    allow_methods=["*"],           # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],           # Allow all headers
)
# --- End CORS Configuration ---


HTML_FILE_PATH = "index.html" # Or index.html

# --- Mount static directories (keep as before) ---
app.mount("/scripts", StaticFiles(directory="scripts"), name="scripts")
app.mount("/styles", StaticFiles(directory="styles"), name="styles")

# --- Keep your route handlers (@app.get("/"), @app.post("/api/process_topology/"), etc.) ---
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open(HTML_FILE_PATH, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{HTML_FILE_PATH} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading HTML file: {str(e)}")

@app.post("/api/process_topology/")
async def api_process_topology(
    max_honeypots: int = Form(3),
    file: UploadFile = File(...)
    ):
    # ... (keep existing implementation) ...
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    try:
        csv_content_bytes = await file.read()
        csv_content = csv_content_bytes.decode('utf-8')

        # --- Add a check for empty content AFTER decoding ---
        if not csv_content or csv_content.isspace():
             print("Warning: Received empty or whitespace-only CSV content.")
             # You might want to return an error here instead of proceeding
             raise HTTPException(status_code=400, detail="Uploaded CSV content is empty.")
        # --- End check ---

        results = backend_logic.process_topology_data(csv_content, max_honeypots)
        return results
    except HTTPException as http_exc: # Re-raise specific HTTP exceptions
         raise http_exc
    except Exception as e:
        print(f"Error processing topology: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing topology: {str(e)}")


@app.get("/api/sample_topology/")
async def api_get_sample_topology(max_honeypots: int = 3):
    # ... (keep existing implementation) ...
     try:
          sample_data_csv = """SystemA,192.168.0.1,192.168.0.2;192.168.0.3,10,host,Windows 7,80;443;3389
SystemB,192.168.0.2,192.168.0.1;192.168.0.4,20,host,Ubuntu 20.04,22;80
SystemC,192.168.0.3,192.168.0.1,5,switch,Switch Firmware,23
SystemD,192.168.0.4,192.168.0.2;192.168.0.5,15,host,Windows 10,445;135
SystemE,192.168.0.5,192.168.0.4;192.168.0.6;192.168.0.7,30,host,Ubuntu 18.04,21;22;80
SystemF,192.168.0.6,192.168.0.5,5,switch,Switch Firmware,
SystemG,192.168.0.7,192.168.0.5;192.168.0.8,25,host,CentOS 7,22;8080
SystemH,192.168.0.8,192.168.0.7;192.168.0.9,10,host,Windows XP,139;445
SystemI,192.168.0.9,192.168.0.8;192.168.0.10,10,host,Ubuntu 20.04,22
SystemJ,192.168.0.10,192.168.0.9;192.168.0.1,20,host,Windows 10,80;443"""
          header = "name,ip,connections,asset_value,device_type,os_type,open_ports\n"
          full_csv = header + sample_data_csv
          results = backend_logic.process_topology_data(full_csv, max_honeypots)
          return results
     except Exception as e:
          print(f"Error processing sample topology: {str(e)}")
          import traceback
          traceback.print_exc()
          raise HTTPException(status_code=500, detail=f"Error processing sample data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # ... (keep uvicorn run command) ...
    print("Starting FastAPI server with CORS enabled...")
    print(f"Serving HTML from: {HTML_FILE_PATH}")
    print("Serving static files from: ./scripts and ./styles")
    print(f"Allowing origins: {origins}")
    print("Access the application at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)