from modelclass import model
from modelwidget_input import make_widget, updatewidget
from modelhelp import build_sorted_bond_desc_dict, build_sorted_rate_desc_dict

from IPython.display import display, HTML, Javascript
import html


def show_heading_text(heading, text):
    heading = html.escape(str(heading))
    text = html.escape(str(text)).replace("\n", "<br>")

    html_block = f"""
    <div style="
        padding: 16px;
        border-radius: 10px;
        background: #f8f9fa;
        border: 1px solid #ddd;
        margin: 10px 0;
    ">
        <h1 style="
            font-size: 36px;
            margin: 0 0 10px 0;
            color: #222;
        ">{heading}</h1>
        <div style="
            font-size: 18px;
            color: #444;
            line-height: 1.5;
        ">{text}</div>
    </div>
    """
    display(HTML(html_block))




def ui_simulations(hide_above=True, hide_mode="cell"):

    mport, baseline = model.modelload('model/port', run=1)

    share_names = mport['new_issue_sh*'].names
    sharedict = build_sorted_bond_desc_dict(share_names)

    sharedef = {
        d: {
            'var': v,
            'value': float(baseline.loc[2026, v]),
            'min': 0,
            'max': 100.0,
            'step': 5.0
        }
        for v, d in sharedict.items()
    }
    sharedef['1 year domestic bond']['slack'] = True
    wsharedef = [
        'sumslide',
        {'content': sharedef, 'maxsum': 100.0, 'heading': 'Issuance policy, shares of each bond'}
    ]

    rate_names = mport['interest_rate__? interest_rate__?? interest_rate_???__? interest_rate_???__?? '].names
    ratedict = build_sorted_rate_desc_dict(rate_names)

    ratedef = {
        d: {
            'var': v,
            'value': float(baseline.loc[2026, v]),
            'min': 0,
            'max': 10.0,
            'step': 0.1
        }
        for v, d in ratedict.items()
    }
    wratedef = ['slide', {'content': ratedef, 'maxsum': 100.0, 'heading': 'Interest rate'}]

    deficit_df = baseline.loc[:, ['DEFICIT']]
    deficitdef = {'update_df': deficit_df}
    wdeficitdef = ['sheet', {'content': deficitdef, 'heading': 'Deficit, update to size in this scenario'}]

    scenariodef = [
        'tab',
        {'content': [
            ['Markets', wratedef],
            ['Issuance', wsharedef],
            ['Deficit', wdeficitdef]
        ],
         'tab': True}
    ]

    wscenario = make_widget(scenariodef)

    mport.var_groups = {
        'Total': 'stock_ultimo__all INTEREST_PAYMENTS__ALL',
        'Interest rates': ' '.join(ratedict.keys()),
        'Issurance': ' '.join(sharedict.keys()),
        'All': '*'
    }
    
    show_heading_text(
        'Debt simulation experimet',
        'This is a small model for simulation debt developements'
    )
    return updatewidget(mport, wscenario)