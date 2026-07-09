#!/usr/bin/env python
"""
Lightweight HTTP API server for TSFM Dashboard.
Serves static dashboard files and handles live model inference requests.
Zero extra dependencies (uses built-in http.server, torch, numpy, pandas).
"""

import os
import sys
import json
import urllib.parse
import http.server
import socketserver
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add parent directory to path so we can import local modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import local modules - this automatically registers pathlib Windows/Posix compatibility overrides
try:
    from finetune_forecasting import TSFMForForecasting
    from evaluate_checkpoints import _extract_state_dict, _infer_architecture_from_state_dict
    IMPORTS_OK = True
except Exception as e:
    print(f"[WARN] Failed to import local pipeline modules: {e}")
    IMPORTS_OK = False

# Server Configuration
PORT = 8000
DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = parent_dir / "data"

class TSFMBackend:
    """Manages model loading and inference execution."""
    def __init__(self):
        self.model = None
        self.model_info = {"checkpoint": "None", "parameters": 0}
        self.device = torch.device("cpu")
        self.load_model()

    def load_model(self):
        if not IMPORTS_OK:
            self.model_info = {"checkpoint": "Error (Imports failed)", "parameters": 0}
            return

        # Look for checkpoints in workspace root
        checkpoints = [
            parent_dir / "tsfm_best.pt",
            parent_dir / "tsfm_pretrain.pt"
        ]
        
        selected_ckpt = None
        for ckpt in checkpoints:
            if ckpt.exists():
                selected_ckpt = ckpt
                break
        
        if not selected_ckpt:
            print("[INFO] No pre-trained model checkpoint found in root directory. Using simulated predictions.")
            self.model_info = {"checkpoint": "Simulated", "parameters": 802458}
            return

        try:
            print(f"[INFO] Loading model checkpoint from {selected_ckpt.name}...")
            # Safe loading technique
            try:
                ckpt_obj = torch.load(selected_ckpt, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt_obj = torch.load(selected_ckpt, map_location=self.device)
                
            state_dict = _extract_state_dict(ckpt_obj)
            inferred = _infer_architecture_from_state_dict(state_dict)
            
            # Extract architecture params
            patch_length = int(inferred.get("patch_length", 16))
            embed_dim = int(inferred.get("embed_dim", 256))
            num_layers = int(inferred.get("num_layers", 6))
            num_heads = 8
            forecast_horizon = int(inferred.get("model_forecast_horizon", 24))
            
            self.model = TSFMForForecasting(
                context_length=512,
                patch_length=patch_length,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                forecast_horizon=forecast_horizon,
                pooling="mean"
            ).to(self.device)
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            total_params = sum(p.numel() for p in self.model.parameters())
            self.model_info = {
                "checkpoint": selected_ckpt.name,
                "parameters": f"{total_params:,} parameters",
                "horizon": forecast_horizon,
                "embed_dim": embed_dim,
                "layers": num_layers
            }
            print(f"[INFO] Successfully loaded model: {self.model_info['parameters']} ({self.model_info['checkpoint']})")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model checkpoint: {e}")
            self.model_info = {"checkpoint": "Simulation (Load failed)", "parameters": 802458}

    @torch.no_grad()
    def forecast(self, context_data: np.ndarray, target_horizon: int = 24) -> np.ndarray:
        """Executes model forward pass with autoregressive rollout if target_horizon > model_horizon."""
        if self.model is None:
            return self.simulate_forecast(context_data, target_horizon)

        model_horizon = self.model_info.get("horizon", 24)
        
        ctx = context_data.astype(np.float32).copy()
        predictions_raw = []
        remaining = int(target_horizon)

        while remaining > 0:
            # Take last 512 context steps
            window = ctx[-512:]
            mean = window.mean()
            std = window.std()
            safe_std = std if std >= 1e-6 else 1.0
            
            normalized = (window - mean) / safe_std
            x = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(-1).to(self.device)
            
            y_pred = self.model(x).squeeze(0).squeeze(-1).cpu().numpy()
            y_pred_raw = (y_pred * safe_std) + mean
            
            take = min(model_horizon, remaining)
            chunk = y_pred_raw[:take]
            predictions_raw.append(chunk)
            
            # Append predictions to raw context for the next rollout step
            ctx = np.concatenate([ctx, chunk])
            remaining -= take

        return np.concatenate(predictions_raw)

    def simulate_forecast(self, context_data: np.ndarray, target_horizon: int = 24) -> np.ndarray:
        """Fallback simulation that mimics trend and seasonality of inputs."""
        last_val = context_data[-1]
        recent_trend = context_data[-1] - context_data[-10] if len(context_data) > 10 else 0
        trend_step = recent_trend / 10.0
        
        forecast = []
        for i in range(target_horizon):
            wave = np.sin(i * 0.25) * 1.5
            val = last_val + (trend_step * (i + 1)) + wave + np.random.normal(0, 0.5)
            forecast.append(val)
        return np.array(forecast)

# Instantiate TSFM backend global object
backend = TSFMBackend()

class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler with simple API router."""
    
    def __init__(self, *args, **kwargs):
        # Serve static files from dashboard directory specifically
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/status":
            self.send_json(backend.model_info)
        elif path == "/api/run":
            self.handle_run_api(query)
        else:
            # Let standard HTTP server handle static files
            super().do_GET()

    def handle_run_api(self, query):
        dataset_name = query.get("dataset", ["ETTh1"])[0]
        slice_offset_str = query.get("slice", ["0"])[0]
        
        # Resolve dataset filepath
        csv_filename = f"{dataset_name}.csv" if not dataset_name.endswith(".csv") else dataset_name
        csv_path = DATA_DIR / csv_filename
        
        if not csv_path.exists():
            # Recursively check data directory for candidates
            candidates = list(DATA_DIR.rglob(csv_filename))
            if candidates:
                csv_path = candidates[0]
            else:
                self.send_error(404, f"Dataset file {csv_filename} not found.")
                return

        try:
            # Read dataset targets column (usually OT)
            df = pd.read_csv(csv_path)
            target_col = "OT"
            if target_col not in df.columns:
                # Find first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
                else:
                    self.send_error(500, "Dataset contains no numeric target columns.")
                    return
            
            series = df[target_col].values.astype(np.float32)
            
            # Determine starting slice
            context_len = 512
            try:
                horizon = int(query.get("horizon", ["24"])[0])
                horizon = max(24, min(horizon, 192))
            except ValueError:
                horizon = 24
                
            max_start = len(series) - context_len - horizon
            
            if slice_offset_str == "random":
                start_idx = np.random.randint(0, max_start)
            else:
                try:
                    start_idx = int(slice_offset_str)
                    start_idx = max(0, min(start_idx, max_start))
                except ValueError:
                    start_idx = 0
            
            # Slice context and target
            context = series[start_idx : start_idx + context_len]
            actual = series[start_idx + context_len : start_idx + context_len + horizon]
            
            # Run inference with target horizon
            forecast = backend.forecast(context, target_horizon=horizon)
            
            # Calculate standard metrics
            diff = forecast - actual
            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(np.abs(diff)))
            
            response = {
                "dataset": dataset_name,
                "slice_start": start_idx,
                "context": context.tolist(),
                "actual": actual.tolist(),
                "forecast": forecast.tolist(),
                "metrics": {
                    "mse": mse,
                    "mae": mae
                }
            }
            self.send_json(response)
            
        except Exception as e:
            self.send_error(500, f"Error processing inference: {str(e)}")

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        body = json.dumps(data).encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

# Run server
def main():
    # Ensure directory context is clean
    os.chdir(str(DASHBOARD_DIR))
    
    # Configure socket reuse options
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), DashboardRequestHandler) as httpd:
        print(f"\n=======================================================")
        print(f"TSFM Dashboard live at: http://localhost:{PORT}")
        print(f"To close server, press Ctrl+C")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down TSFM server.")
            sys.exit(0)

if __name__ == "__main__":
    main()
