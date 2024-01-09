If you have not checked out the submodules:
```
git submodule update --init --recursive
```

This also requires the `rl_tools/tests/data` submodule to be checked out from from the `rl_tools` repo (since the demo NN checkpoint is included from `../../../tests/data/nn_models_sequential_persist_code`):
```
git submodule update --init -- tests/data
```

Build using `make cf2_defconfig; make clean; make -j12`

Upload using `CLOAD_CMDS="-w radio://0/80/2M" make cload`


Once uploaded the cf will do the calculations but once you connect to it using cfclient it might crash (because the computation blocks the communication task for too long). But it still receives the logs from before the crash so you can evaluate the runtimes. 