# Third-party software

Cueframe's source and code-generated showcase media are offered under the MIT license in
`LICENSE`. No stock footage, recorded voices, external music, model weights, or downloaded
artwork are included. Text in the demo is invented scene annotation.

The package uses these separately installed dependencies:

| Software | License / role |
| --- | --- |
| NumPy | BSD 3-Clause; array and signal processing |
| Pillow | HPND / historical PIL license; raster drawing and bundled default font |
| FFmpeg / ffprobe | External executables, not bundled. Licensing depends on build options; the render path requires an FFmpeg build with libx264, generally GPL-enabled. |
| pytest | MIT; development tests only |
| Ruff | MIT; development lint only |

The dependency distributions retain their own license notices. Redistribution of a bundled
environment or FFmpeg binary requires checking that distribution's terms. Cueframe invokes
the executables locally and does not ship them.
