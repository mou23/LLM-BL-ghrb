import pyflagr.Linear as Linear
import pandas as pd

def combmnz(input_file,out_dir,merge_type):
    if merge_type==3 or merge_type==9:
        robust = Linear.CombMNZ(norm='rank')
    elif merge_type==2:
        robust = Linear.CombMNZ(norm='score')
    elif merge_type==1:
        robust = Linear.CombSUM(norm='rank')
    elif merge_type==0:
        robust = Linear.CombSUM(norm='score')

    df_out,_ = robust.aggregate(input_file=input_file,out_dir=out_dir)
    # df_out.to_csv(f"{out_dir}/candidate.csv",index=False,header=False)
    # indexes = list(set(df_out.index.tolist()))
    # candidate_list = {}
    # for ind in indexes:
    #     df = df_out.loc[ind,'ItemID']
    #     try:
    #         candidate_list[ind] = df.tolist()
    #     except:
    #         candidate_list[ind] = [df]
    top_k = 20
    candidate_list = {}
    for query_id in df_out['Query'].unique():
        sub_df = df_out[df_out['Query'] == query_id]
        top_items = sub_df.sort_values(by='Score', ascending=False).head(top_k)['ItemID'].tolist()
        candidate_list[query_id] = top_items

    return candidate_list

# if __name__ == '__main__':
#     candidate_ids = combmnz("../expresults/aspectj/raw/aspectj_candidate.csv","../expresults/tmp",9)
#     print(candidate_ids)