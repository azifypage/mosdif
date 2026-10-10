import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from cli import main
        main()
    else:
        try:
            from mosaicdiff.app import run
            run()
        except Exception:
            # Fallback for headless environments without DISPLAY
            from cli import main
            main()
