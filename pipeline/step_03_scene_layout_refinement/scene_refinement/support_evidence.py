"""Static, labeled collision-geometry evidence for support selection."""
from pathlib import Path

import numpy as np

from .protocol import digest, file_hash, write_json


def diagram(request, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    geometry = request['diagram_triangles_world']
    catalog = {k:v for k,v in request.items() if k != 'diagram_triangles_world'}
    binding = {'catalog_sha256':digest(catalog),'geometry_sha256':digest(geometry),
               'implementation_sha256':file_hash(__file__), 'kind':'support_layer_diagram'}
    path = directory / (digest(binding)+'.png')
    fig, axes = plt.subplots(1,3,figsize=(18,7),gridspec_kw={'width_ratios':[1,1,1.4]})
    options = catalog['options']; colors = plt.get_cmap('tab20')
    for index, option in reversed(list(enumerate(options))):
        triangles = np.asarray(geometry[option['region_id']])
        color = colors(index % 20)
        axes[0].add_collection(PolyCollection(triangles[:,:,:2],facecolors=[color],alpha=.65,linewidths=0))
        axes[1].add_collection(PolyCollection(triangles[:,:,[0,2]],facecolors=[color],alpha=.65,linewidths=0))
        low, high = np.asarray(option['bounds_world_m'])
        axes[1].plot([low[0],high[0]],[high[2],high[2]],color=color,lw=1)
    low, high = np.asarray(catalog['child_bounds_world_m'])
    for ax, dims, title in [(axes[0],[0,1],'Top view: actual support triangles'),
                            (axes[1],[0,2],'Side view: actual support triangles')]:
        ax.add_patch(Rectangle(low[dims],*(high-low)[dims],fill=False,edgecolor='black',ls='--',lw=1.5))
        ax.autoscale();ax.set_aspect('equal',adjustable='datalim');ax.set_title(title)
        ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)' if dims[1]==1 else 'Z (m)')
    axes[2].axis('off')
    text = ['Layer ID / actual component names / top height', '']
    for option in options:
        text.append(f"{option['diagram_label']}  {', '.join(option['component_names'])}\n"
                    f"     {option['height_world_m']:.6f} m  |  {option['region_id']}")
    axes[2].text(0,1,'\n'.join(text),va='top',family='monospace',fontsize=max(5,min(9,180/max(1,len(options)))))
    fig.suptitle(f"{catalog['child_object_id']} on {catalog['parent_object_id']} | dashed box: child input bounds\n"
                 'Geometry proposal evidence; neither a photograph nor a physical acceptance result')
    fig.tight_layout();fig.savefig(path,dpi=100);plt.close(fig)
    receipt = {'path':str(path.resolve()),'sha256':file_hash(path),'derivation':binding}
    write_json(path.with_suffix('.json'),receipt)
    return catalog, receipt
