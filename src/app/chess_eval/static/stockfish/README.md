# Stockfish 19 browser build

Vendored from the `@lichess-org/stockfish-web` npm package, version `0.6.1`.

- Upstream: <https://github.com/lichess-org/stockfish-web>
- Build: `sf_19.js` + `sf_19.wasm` (Stockfish 19)
- NNUE: `nn-1a298aa575a0.nnue`, downloaded and cached by the app at
  `data/models/stockfish/nn-1a298aa575a0.nnue`
- License: the upstream package is AGPL-3.0-or-later; the vendored LICENSE file
  is included alongside the build.

The engine runs in a Web Worker. The app serves these files with
cross-origin-isolation headers so the threaded WebAssembly build can use
`SharedArrayBuffer`.
