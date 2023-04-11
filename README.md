Build using `make cf2_defconfig; make clean; make -j12`

Upload using `CLOAD_CMDS="-w radio://0/80/2M" make cload`


Once uploaded the cf will do the calculations but once you connect to it using cfclient it might crash (because the computation blocks the communication task for too long). But it still receives the logs from before the crash so you can evaluate the runtimes. 