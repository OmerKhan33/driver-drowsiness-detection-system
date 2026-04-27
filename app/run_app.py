"""
Ngrok launcher for Driver Drowsiness Detection Streamlit app.

Starts the Streamlit server and creates an ngrok tunnel so the app
can be accessed remotely via a public URL.

Usage:
    python app/run_app.py --ngrok-token YOUR_NGROK_AUTH_TOKEN
    python app/run_app.py --ngrok-token YOUR_TOKEN --port 8501
"""

import argparse
import subprocess
import sys
import time


def main():
    """Launch Streamlit app with ngrok tunnel."""
    parser = argparse.ArgumentParser(
        description="Launch Drowsiness Detection app with ngrok tunnel"
    )
    parser.add_argument(
        "--ngrok-token",
        type=str,
        required=True,
        help="Your ngrok auth token (from https://dashboard.ngrok.com)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Local port for Streamlit (default: 8501)",
    )
    args = parser.parse_args()

    # Set ngrok auth token
    try:
        from pyngrok import conf, ngrok

        conf.get_default().auth_token = args.ngrok_token
    except ImportError:
        print("ERROR: pyngrok not installed. Run: pip install pyngrok")
        sys.exit(1)

    # Start Streamlit in background
    print(f"\n  Starting Streamlit on port {args.port}...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
    ]
    proc = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Streamlit to start
    time.sleep(5)

    # Create ngrok tunnel
    print("  Creating ngrok tunnel...")
    try:
        public_url = ngrok.connect(args.port, "http")
        print("\n" + "=" * 60)
        print("  🚗 Driver Drowsiness Detection System")
        print("=" * 60)
        print(f"\n  Local URL:  http://localhost:{args.port}")
        print(f"  Public URL: {public_url}")
        print("\n  Share the public URL to access remotely!")
        print("  Press Ctrl+C to stop.\n")
        print("=" * 60)

        # Keep running
        proc.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
    except Exception as e:
        print(f"\n  Ngrok error: {e}")
        print("  Check your auth token and try again.")
    finally:
        ngrok.kill()
        proc.terminate()
        print("  Done.")


if __name__ == "__main__":
    main()
