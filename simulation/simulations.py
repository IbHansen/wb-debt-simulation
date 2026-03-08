
from modelclass import model 
from modelwidget_input import  make_widget,updatewidget
from modelhelp import build_sorted_bond_desc_dict,build_sorted_rate_desc_dict




# In[4]:
def ui_simulations():
    mport,baseline = model.modelload('model/port',run=1)
    share_names = mport['new_issue_sh*'].names
    sharedict = build_sorted_bond_desc_dict(share_names)
    
    
    # In[5]:
    
    
    sharedef = {d: 
                {'var':v,
                'value' : float(baseline.loc[2026,v  ]),
                 'min':0,
                 'max':100.0 ,'step':5.0
                } for v,d in sharedict.items()}
    sharedef['1 year domestic bond']['slack']=True 
    wsharedef =['sumslide', {'content':sharedef,'maxsum':100.0,'heading':'Issuance policy, shares of each bond'}]
    
    
    # In[6]:
    
    
    rate_names= mport['interest_rate__? interest_rate__?? interest_rate_???__? interest_rate_???__?? '].names
    ratedict = build_sorted_rate_desc_dict(rate_names)
    
    ratedef = {d: 
                {'var':v,
                'value' : float(baseline.loc[2026,v  ]),
                 'min':0,
                 'max':10.0 ,'step':0.1
                } for v,d in ratedict.items()}
    wratedef = ['slide',{'content':ratedef,'maxsum':100.0,'heading':'Interest rate'}]
    
    
    # In[7]:
    
    
    deficit_df = baseline.loc[:,['DEFICIT']]
    
    deficitdef = {'update_df':deficit_df}
    wdeficitdef = ['sheet',{'content':deficitdef,'heading':'Deficit, update to size in this scenario'}]
    wdeficit = make_widget(wdeficitdef)
    
    
    # In[8]:
    
    
    scenariodef = ['tab',{'content':[
                                  ['Markets',wratedef],
                                  ['Issuance' ,wsharedef],
                                  ['Deficit' ,wdeficitdef]
                                ]
                                ,'tab':True}]
    
    
    wscenario=make_widget(scenariodef)
    
    
    # In[9]:
    
    
    mport.var_groups={'Total'          :'stock_ultimo__all INTEREST_PAYMENTS__ALL', 
                      'Interest rates' : ' '.join(ratedict.keys()) ,
                      'Issurance'      :  ' '.join(sharedict.keys()), 
                      'All'            :  '*'}
    
    
    # In[10]:
    
    
    return updatewidget(mport,wscenario)

