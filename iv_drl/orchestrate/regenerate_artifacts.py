from iv_drl.orchestration.run_pipeline import run_pipeline as main

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild all artifacts via pipeline driver")
    ap.add_argument("--force", action="store_true", help="Delete existing outputs before rebuilding")
    args = ap.parse_args()
    main(force=args.force) 