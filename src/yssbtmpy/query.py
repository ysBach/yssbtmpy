from astroquery.jplsbdb import SBDB

__all__ = ["get_aphel_jd", ]


def get_aphel_jd(objid: str | dict[str, float | str], plus: bool = True) -> float:
    """ Calculate the epoch of aphelion based on the object id thru SBDB

    Parameters
    ----------
    objid : str or dict
        Name, number, or designation of target object. Uses the same codes
        as JPL Horizons. Arbitrary topocentric coordinates can be added in
        a dict. The dict has to be of the form {``'lon'``: longitude in deg
        (East positive, West negative), ``'lat'``: latitude in deg (North
        positive, South negative), ``'elevation'``: elevation in km above
        the reference ellipsoid, [``'body'``: Horizons body ID of the
        central body; optional; if this value is not provided it is assumed
        that this location is on Earth]}.

    plus: bool, optional
        Whether to return the epoch of aphelion as perihelion plus "orbital
        period/2" (default). Otherwise, it is minus half the orbital period.
    """
    orb_elem = SBDB.query(objid, full_precision=False)["orbit"]["elements"]
    per_orb = orb_elem["per"].si.value  # in seconds
    tp = orb_elem["tp"].value  # in JD [days]
    return tp + per_orb/2/86400 if plus else tp - per_orb/2/86400


