# Release and submission checklist

- [x] Create a new private repository, without importing unrelated Git history.
- [x] Preserve original experiment files and export actual recorded predictions/votes.
- [x] Execute offline score/statistical reproduction and asset checks; results are in `verification/`.
- [ ] Authors review the package and the verification records.
- [x] The repository owner authorized renaming the repository to `SecPrompt-Transfer` and making it public on 7 November 2026.
- [ ] Select a license for original research code, if broader reuse permissions are to be granted.
- [x] Retain third-party attribution and existing component licenses. The archived GitReq v3 record specifies CC BY 4.0; live Figshare access returned HTTP 403 during the release check.
- [ ] Confirm author names, funding, competing interests and contribution statements in the manuscript.
- [x] Publish `2206113885/SecPrompt-Transfer` and update the manuscript's public-availability link on 7 November 2026.
- [x] Verify repository metadata and README access through GitHub's API without authentication (`private: false`).
- [ ] Create a tagged release; optionally deposit it in a persistent archive and add the resulting DOI (none is currently assigned).

The manuscript is not included here as a separate outdated draft. Its authoritative text remains in the linked editing project. Model weights, credentials, SSH configuration, private conversation logs and complete local/server backups must not be added to this repository.
