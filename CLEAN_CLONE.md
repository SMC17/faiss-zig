# Clean clone

`zig build test` does not succeed from this repository alone.

## Sibling path-deps

| In-tree path | Sibling | Local layout | Why it is required |
|---|---|---|---|
| `deps/qsr-zig` | `SMC17/qsr-zig` | workstation symlink `deps/qsr-zig` → `../../qsr-zig`, or a checkout at `deps/qsr-zig` | `build.zig` always imports `deps/qsr-zig/src/root.zig` as module `qsr` (used by `src/topological_vector.zig`). `deps/*` is gitignored except `deps/.gitkeep`. |

`build.zig.zon` has `.dependencies = .{}` — the sibling is a `b.path(...)` source tree, not a hashed Zig package.

`tokenizers-zig` is a lodestone sibling with no path-deps (its `build.zig.zon` is empty). It is not required to build this tree.

## CROSS_REPO_PAT

Default `GITHUB_TOKEN` cannot read other private repositories. CI
(`.github/workflows/ci.yml`, `hygiene-zig.yml`) checks `SMC17/qsr-zig`
into `deps/qsr-zig` with:

```yaml
token: ${{ secrets.CROSS_REPO_PAT || github.token }}
```

Set repository secret `CROSS_REPO_PAT` to a PAT with `contents:read` on
`SMC17/qsr-zig`. Until that secret exists, clean-clone CI falls back to
`github.token` and private sibling checkout fails.

## Local bootstrap

```sh
git clone <this-repo>
cd faiss-zig
# either sibling symlink (matches this workstation):
ln -sfn ../../qsr-zig deps/qsr-zig
# or a private checkout into deps/:
#   git clone <qsr-zig-url> deps/qsr-zig
zig build test
```
