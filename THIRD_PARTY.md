# Software and media notices

Cueframe's source code and its synthetic animation and multicamera showcases are offered
under the MIT license. [License](LICENSE)

## Hong Kong #2

The finished Hong Kong #2 excerpt includes vAPI Network footage, recorded speech, and
branded illustrations and captions. The excerpt and its preview still are provided for
viewing and are excluded from the MIT code license. Rights in the media and brand assets
remain with their respective rights holders. The source footage, voice tracks, full
reference film, fonts, LUT and edit records are private.
[Finished excerpt](docs/hk2_demo.mp4) · [Preview still](docs/hk2_preview.png)

## Software dependencies

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
