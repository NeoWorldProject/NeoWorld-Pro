Repair the supplied failed physical mesh copies. Visual geometry and joints are frozen.
Return JSON {"repairs":[{"id":"supplied id","operation":"coacd_remesh","resolution":50,"reason":"..."}]}.
Include every supplied representative ID once; equivalent parts reuse that decision.
Choose only coacd_remesh, meshfix, convex_hull, box, boolean_union or boolean_difference.
For intersecting closed shells, choose union or difference using the operand images;
provide base_operand (zero-based) for difference. A proxy is allowed if neither is usable.
coacd_remesh uses CoACD's distance-field reconstruction with
integer resolution 20..100. meshfix repairs the surface before that reconstruction.
Use the same operation/settings for equivalent failures. Convex hull and box are
explicitly degraded physical proxies and may fill cavities. Choose them when surface
reconstruction cannot retain a usable solid. Do not alter mass, joint state or visual geometry.
