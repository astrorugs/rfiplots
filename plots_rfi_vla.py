import numpy as np
import glob
import pandas 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
import plotly.express as px
from scipy.interpolate import interp1d

############# data ############# 
# read spectra
db = {}
fns = glob.glob('../rfi_scans_vla/*.csv')

for fn in fns:
        # identifier
        name = fn.split('/')[-1].split('.csv')[0]
        db[name] = {}

        # read in spectrum
        spec = pandas.read_csv(fn)

        # channel width

        # bring all data to 100 kHz resolution
        db[name]['band'] = name.split('_')[0]
        if db[name]['band'] == 'U':
                db[name]['band'] = 'Ku'
        elif db[name]['band'] == 'A':
                db[name]['band'] = 'Ka'
        db[name]['pol'] = name.split('_')[1]
        db[name]['date'] = '-'.join(name.split('_')[3:])
        db[name]['freq'] = np.array(spec['Frequency'])
        db[name]['amp'] = np.array(spec['Ampl(Jy)'])
        db[name]['phase'] = np.array(spec['Phase'])

dbp = pandas.DataFrame(db).T
dates = np.unique(dbp['date'])

configs = [['A',['2023-07-30','2023-08-01'],'July 2023, A-config'],
           ['D',['2023-10-25','2023-11-28'],'November 2023, D-config'],
           ['C',['2024-02-14','2024-04-19'],'February 2024, C-config'],
           ['B',['2024-05-08','2024-10-04'],'May 2024, B-config'],
]


############# setup plot ############# 
fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    shared_yaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles = ['Median (all spectra, all polarization)',
                                      '75% Percentile (all spectra, all polarization)',
                                      ],
                    )

# create shuffled colormap
np.random.seed(18)
color_scale = px.colors.sample_colorscale('Turbo',np.random.random(100))


############# plot data #############
# bands
vla = [
        [0.224,0.480,"rgba(0, 0, 255, 0.3)",'P-band'],
        [1.000,2.000,"rgba(138, 43, 226, 0.3)",'L-band'],
        [2.000,4.000,"rgba(165, 42, 42, 0.3)",'S-band'],
        [4.000,8.000,"rgba(222, 184, 135, 0.3)",'C-band'],
        [8.000,12.00,"rgba(95, 158, 160, 0.3)",'X-band'],
        [12.00,18.00,"rgba(127, 255, 0, 0.3)",'Ku-band'],
        [18.00,26.50,"rgba(210, 105, 30, 0.3)",'K-band'],
        [26.50,40.00,"rgba(100, 149, 237, 0.3)",'Ka-band'],
        [40.00,50.00,"rgba(220, 20, 60, 0.3)",'Q-band'],
        ]

for m,config in enumerate(configs):
    sub_dates = [_ for _ in dates if ((datetime.datetime.fromisoformat(config[1][0])<datetime.datetime.fromisoformat(_)) &
                             (datetime.datetime.fromisoformat(config[1][1])>datetime.datetime.fromisoformat(_)))]

    # interpolate all bands and dates to the same frequencies
    for o, _vla in enumerate(vla):
            band = _vla[3].split('-')[0]
            subdpb = dbp[(dbp.band == band)&([_ in sub_dates for _ in dbp.date])].copy()
            subdpb.reindex()
            for n in subdpb.index:
                f = interp1d(subdpb.at[n,'freq'], subdpb.at[n,'amp'], assume_sorted=False, fill_value=(np.nan,np.nan), bounds_error=False)
                subdpb.at[n,'amp']  =  f(subdpb.at[subdpb.index[0],'freq'])/np.median(subdpb.at[n,'amp']) # interpolate and normalize
                subdpb.at[n,'freq'] = subdpb.at[subdpb.index[0],'freq']

            # spectra to plot
            f = np.array(subdpb['freq'])[0]
            amed = np.median(np.vstack(np.array(subdpb['amp'])),axis=0)
            amax = np.percentile(np.vstack(np.array(subdpb['amp'])),75,axis=0)
            
            # break up spectra by spw (searching for uneven frequency jumps)
            fdiff = f[1:] - f[:-1]
            jumps = np.append(0,np.nonzero(fdiff != np.median(fdiff))[0]+1) # register all jumps, add first channel
            juord = np.append(jumps[np.argsort(f[jumps])],len(f)) # re-order jumps, add last channel
            
            for j in range(len(jumps)-1):
                    fig.add_trace(go.Line(x=f[jumps[j]:jumps[j+1]]/1000., y=10.*np.log10(amed[jumps[j]:jumps[j+1]]), line=dict(width=0.5,color=color_scale[m]),  name= config[2] + ' ({} Spectra)'.format(len(sub_dates)) , legendgroup=f'group{m}',showlegend=False),row=1, col=1)
                    fig.add_trace(go.Line(x=f[jumps[j]:jumps[j+1]]/1000., y=10.*np.log10(amax[jumps[j]:jumps[j+1]]), line=dict(width=0.5,color=color_scale[m]),  name= config[2] + ' ({} Spectra)'.format(len(sub_dates)), legendgroup=f'group{m}',showlegend=(o==0) & (j==0)),row=2, col=1)

############## plot VLA bands ############ 
boxvla = []
for m,entry in enumerate(vla):
        boxvla.append(dict(type="rect",
                           x0=entry[0], y0=-120,  
                           x1=entry[1], y1=50,    
                           fillcolor=entry[2],
                           line=dict(color="rgba(255, 255, 255, 0)"),
                           xref='x',yref='y',
                           )
                      )
        fig.add_trace(go.Line(x=[np.mean([entry[0],entry[1]]),np.mean([entry[0],entry[1]])], y=[-120, 50],
                              line=dict(width=0,color=entry[2].replace('0.3','1.0')),
                              hoverlabel = dict(namelength = -1),hovertemplate=entry[3],
                              name= entry[3], showlegend=False),row=1, col=1)
        fig.add_trace(go.Line(x=[np.mean([entry[0],entry[1]]),np.mean([entry[0],entry[1]])], y=[-120, 50],
                              line=dict(width=0,color=entry[2].replace('0.3','1.0')),
                              hoverlabel = dict(namelength = -1),hovertemplate=entry[3],
                              name= entry[3], showlegend=False),row=2, col=1)

############## plot known RFI ############ 
# read RFI
txt = open('rfi_tags.txt','r')
lines = []
for line in txt:
    lines.append(line)
txt.close()
    
lines_s = [_.split('\t') for _ in lines]

# make a 2 MHz width if no width is given
rfi = []
for _ in lines_s:
    if len(_[0].split('-'))==1:
            rfi.append([_[0],float(_[0])-0.1,float(_[0])+0.1,'<br>'.join(_[1:]).replace('plot','').replace('\n','').replace(' ','<br>')])
    else:
            rfi.append([_[0],float(_[0].split('-')[0]),float(_[0].split('-')[1]),'<br>'.join(_[1:]).replace('plot','').replace('\n','').replace(' ','<br>')])


boxrfi = []
for m,entry in enumerate(rfi):
        boxrfi.append(dict(type="rect",
                           x0=entry[1]/1000., y0=-120,  
                           x1=entry[2]/1000., y1=50,    
                           fillcolor="rgba(211, 211, 211, 0.3)",
                           line=dict(color="rgba(255, 255, 255, 0)"),
                           xref='x',yref='y',
                           )
                      )

        fig.add_trace(go.Line(x=[np.mean([entry[1]/1000.,entry[2]/1000.])], y=[50],
                              line=dict(width=0.5,color="rgba(211, 211, 211, 1.0)"),
                              name= entry[0] + ' MHz<br>'+entry[3],  hovertemplate= entry[0] + ' MHz<br>'+entry[3],showlegend=False ,
                              hoverlabel = dict(namelength = -1)),row=1, col=1)
        fig.add_trace(go.Line(x=[np.mean([entry[1]/1000.,entry[2]/1000.])], y=[50],
                              line=dict(width=0.5,color="rgba(211, 211, 211, 1.0)"),
                              name= entry[0] + ' MHz<br>'+entry[3],hovertemplate= entry[0] + ' MHz<br>'+entry[3],  showlegend=False,
                              hoverlabel = dict(namelength = -1)),row=2, col=1)

print(boxrfi)

############## lines ############
# works but makes things really slow
# lines
txt = open('line_list.txt','r')
lines = []
for line in txt:
    lines.append(line)
txt.close()
    
lines_s = [[float(_.split(',')[0]),','.join(_.split(',')[1:]).replace('\t', '<br>'), _.split('\t')[1]] for _ in lines]

linenames = np.unique([_[2] for _ in lines_s])

boxlin = []
np.random.seed(len(linenames))
color_scale = px.colors.sample_colorscale('Jet',np.random.random(len(linenames)))

for l, name in enumerate(linenames):
    for m,entry in enumerate([_ for _ in lines_s if _[2] == name ]):
        boxlin.append(dict(type="rect",
                           x0=entry[0]-0.001, y0=-120,  
                           x1=entry[0]+0.001, y1=50,    
                           fillcolor=color_scale[l],
                           opacity=0.4,
                           line=dict(color="rgba(255, 255, 255, 0)"),
                           xref='x',yref='y',
                           )
                      )
        fig.add_trace(go.Line(x=[entry[0]], y=[45],
                              line=dict(width=0.5, color=color_scale[l]),
                              name= name, legendgroup=f'group{len(dates)+1+l}', showlegend=( (m==0)),legend='legend2',
                              hoverlabel = dict(namelength = -1),hovertemplate=entry[1]),
                      row=1, col=1)
        fig.add_trace(go.Line(x=[entry[0]], y=[45],
                              line=dict(width=0.5, color=color_scale[l]),
                              name= name, legendgroup=f'group{len(dates)+1+l}', showlegend=False,legend='legend2',
                              hoverlabel = dict(namelength = -1),hovertemplate=entry[1]),
                      row=2, col=1)

############## plot layout ############ 
        fig.update_layout(title_text="VLA RFI (csv files from https://science.nrao.edu/facilities/vla/docs/manuals/obsguide/rfi)")

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=1.25,
            buttons=[
                dict(label="None",
                     method="relayout",
                     args=[{"shapes": []}]),
                dict(label="VLA bands",
                    method="relayout",
                    args=[{"shapes": boxvla+[{**_ , 'xref':'x2','yref':'y2'}  for _ in boxvla]}]),
                dict(label="Astrophysical Lines (2 MHz BW)",
                    method="relayout",
                    args=[{"shapes": boxlin+[{**_ , 'xref':'x2','yref':'y2'}  for _ in boxlin]}]),
                dict(label="Known RFI",
                     method="relayout",
                     args=[{"shapes": boxrfi+[{**_ , 'xref':'x2','yref':'y2'} for _ in boxrfi]}]),
            ]
        ),
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=1.3,
            buttons=[
                    dict(label="Linear",  
                         method="relayout", 
                         args=[{"xaxis.type": "linear","xaxis2.type": "linear"}]),
                    dict(label="Log", 
                         method="relayout", 
                         args=[{"xaxis.type": "log","xaxis2.type": "log"}]),            ]
        )
    ]
)

# labels
fig.update_layout(
        {
                "yaxis" : {'title':'10 * log10 (F [normalized])'},
                "yaxis2" : {'title':'10 * log10 (F [normalized])'},
                "xaxis2" : {'title':'Frequency [GHz]'},
        }
)

# axis ranges
fig.update_layout(
        {
                "xaxis": {"range": [0,50]},
                "yaxis": {"range": [-20,55]},
                "yaxis2": {"range": [-20,55]},
                "legend" : dict(
                        orientation='v',
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=1.2,
                        font=dict(size=10),
                        tracegroupgap=2,
                ),
                "legend2" : dict(
                       orientation='v',
                       yanchor="top",
                       y=0.3,
                       xanchor="right",
                       x=1.2,
                       font=dict(size=10),
                       tracegroupgap=2,
                ),
        }
)



fig.update_layout(
        {
                'plot_bgcolor': 'rgba(255, 255, 255, 100)',
                'paper_bgcolor' : 'rgba(255, 255, 255, 100)',
                'font':dict(family='Times New Roman',size=12),
                "legend_font_color":'black',
                'autosize':False,
        }
)
for _ax in ['xaxis','yaxis','xaxis2','yaxis2',]:
        fig.update_layout(
                {
                        _ax : dict(mirror=True,
                                   ticks='outside',
                                   showline=True, 
                                   showgrid=False, 
                                   color='black', 
                                   linecolor='black'),
                }
        )
        



config = config = {'scrollZoom': True,
                   'responsive': True}
fig.show(config=config)
                      
fig.update_layout(autosize = True,)
fig.write_html("plot_rfi_vla.html",include_plotlyjs=True,config=config)

fig.update_layout(height=800, width=1600,)
fig.write_image("plot_rfi_vla.pdf")
