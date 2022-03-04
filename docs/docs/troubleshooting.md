!!! fail

    I have `8` CPUs but `pandarallel` uses by default only `4` workers, and there is no
    performance increase (and maybe a little performance decrease) if I manually set the
    number of workers to a number higher than `4`.

`pandarallel` can only speed up computation until about the number of **cores** your
computer has. The majority of recent CPUs (like Intel Core i7) uses hyperthreading.
For example, a 4-core hyperthreaded CPU will show 8 CPUs to the operating system, but
will **really** have only 4 physical computation units.

On **Ubuntu**, you can get the number of cores with

```bash
grep -m 1 'cpu cores' /proc/cpuinfo
```

!!! fail

    I use `jupyter lab` and instead of progress bars, I see these kind of things:
    ```
    VBox(children=(HBox(children=(IntProgress(value=0, description='0.00%', max=625000), Label(value='0 / 625000')
    ```

Run the following 3 lines, and you should be able to see the progress bars:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

(You may also have to install `nodejs` if asked)
