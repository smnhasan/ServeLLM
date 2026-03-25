"""
llm_serve.cli — Command-line interface.

Usage examples
--------------
Start server (blocking keep-alive):
    llm-serve start --authtoken YOUR_TOKEN

Start without keep-alive (returns immediately — useful for testing):
    llm-serve start --authtoken YOUR_TOKEN --no-keep-alive

Override port or GPU settings:
    llm-serve start --authtoken YOUR_TOKEN --port 8001 --n-gpu-layers 0
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-serve",
        description="OpenAI-compatible API server for GPT-OSS-20B + embedding models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── start ──────────────────────────────────────────────────────────────
    start = sub.add_parser("start", help="Download models and start the API server.")

    start.add_argument(
        "--authtoken", "-t",
        default=None,
        help="Ngrok auth token (or set NGROK_AUTHTOKEN env var).",
    )
    start.add_argument(
        "--port", "-p",
        type=int, default=8000,
        help="Local uvicorn port (default: 8000).",
    )
    start.add_argument(
        "--max-hours",
        type=int, default=12,
        help="Keep-alive duration in hours (default: 12).",
    )
    start.add_argument(
        "--n-ctx",
        type=int, default=10_048,
        help="LLM context window size (default: 10048).",
    )
    start.add_argument(
        "--n-gpu-layers",
        type=int, default=-1,
        help="GPU layers to offload (-1 = all, 0 = CPU only, default: -1).",
    )
    start.add_argument(
        "--max-requests",
        type=int, default=3,
        help="Max concurrent requests (default: 3).",
    )
    start.add_argument(
        "--llm-repo",
        default="ggml-org/gpt-oss-20b-GGUF",
        help="HuggingFace repo id for the GGUF model.",
    )
    start.add_argument(
        "--llm-file",
        default="gpt-oss-20b-mxfp4.gguf",
        help="GGUF filename inside the repo.",
    )
    start.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Return immediately instead of blocking (useful for testing).",
    )

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.command == "start":
        if args.no_keep_alive:
            from llm_serve.launch import start_server_only
            manager, _ = start_server_only(
                llm_repo=args.llm_repo,
                llm_file=args.llm_file,
                n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers,
                max_requests=args.max_requests,
                port=args.port,
                authtoken=args.authtoken,
            )
            if manager is None:
                return 1
            print(f"Server running at: {manager.get_public_url()}")
            print("(No keep-alive — process will exit.)")
        else:
            from llm_serve.launch import start_server_and_keep_alive
            start_server_and_keep_alive(
                llm_repo=args.llm_repo,
                llm_file=args.llm_file,
                n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers,
                max_requests=args.max_requests,
                port=args.port,
                authtoken=args.authtoken,
                max_hours=args.max_hours,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
