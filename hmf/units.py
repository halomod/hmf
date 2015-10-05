import astropy.units as u

h_unit = u.def_unit("h")

m_unit = u.MsolMass/h_unit
r_unit = u.Mpc/h_unit
k_unit = h_unit/u.Mpc
rho_unit = m_unit/r_unit**3
