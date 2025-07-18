# yssbtmpy

My personal Python package to do thermal modeling on atmosphereless bodies in the Solar system.

## Citation
Currently, there is no proper publication deals with the details of the implementation of this package (because it is nothing but a collection of the works by other people, although I implemented them).

If this package was useful for your work, please consider citing [Bach & Ishiguro (2021) A&A](https://ui.adsabs.harvard.edu/abs/2021A%26A...654A.113B/abstract). BibTeX:
```
@ARTICLE{2021A&A...654A.113B,
       author = {{Bach}, Yoonsoo P. and {Ishiguro}, Masateru},
        title = "{Thermal radiation pressure as a possible mechanism for losing small particles on asteroids}",
      journal = {\aap},
     keywords = {minor planets, asteroids: general, asteroids: individual: 3200 Phaethon, meteorites, meteors, meteoroids, interplanetary medium, Astrophysics - Earth and Planetary Astrophysics},
         year = 2021,
        month = oct,
       volume = {654},
          eid = {A113},
        pages = {A113},
          doi = {10.1051/0004-6361/202040151},
archivePrefix = {arXiv},
       eprint = {2108.03898},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...654A.113B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

This package has been used by:
* [Bach & Ishiguro (2021) A&A](https://ui.adsabs.harvard.edu/abs/2021A%26A...654A.113B/abstract)
* [Beniyama et al. (2022) PASJ](https://ui.adsabs.harvard.edu/abs/2022PASJ...74..877B/abstract) (priv. comm.)

## Notes
Unit of `flam` ($F_\lambda$) used by astronomical packages unfortunately is a strange cgs-style unit, erg/s/cm2/Å. In this package, we use a more standard unit FLAM_SI = W/m2/µm, following, for example, solar standard spectrum (ASTM 2000), engineering and physics fields, and spacecraft/telescope designs. Conversion is:
  1 FLAM_SI = 10 FLAM_ASTRONOMY (if your flux in classical astronomical FLAM, divide it by 10 to get FLAM_SI)
For `fnu` ($F_\nu$), we use Jy, which is widely used in radio sciences.

**Why "ys"sbtmpy? The name "sbtmpy" is too general, and I believe a better package should take that name, so I decided not to occupy the name. I see many useless packages that preoccupy meaningful names...**


