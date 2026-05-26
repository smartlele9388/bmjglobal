from __future__ import annotations

import step6_submission_ready_statistics as stats
import step6_submission_ready_figures as figs


def main() -> None:
    stats.main()
    figs.main()
    stats.quality_log(patient_found=True)
    zip_path = stats.create_requested_zip()
    print(f"Step 6 complete. Requested bundle: {zip_path}")


if __name__ == "__main__":
    main()
