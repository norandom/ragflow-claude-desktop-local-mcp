# Repository workflow

- Work directly on `main`. The owner explicitly requested one branch and no
  feature or release branches for routine work.
- Validate changes, use SSH-signed commits, and push to `main` when authorized.
- Tag releases from the validated `main` commit and verify the release workflow.
- Incorporate dependency updates without downgrading newer fixes already on
  `main`; remove superseded bot branches when cleanup is authorized.
