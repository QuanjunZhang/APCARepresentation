import pickle

feature_data="/Users/tom/Downloads/learning-program-representation-master/data/custom/feature/data.pkl"
sequence_data="/Users/tom/Downloads/learning-program-representation-master/data/custom/feature/fusion/data.pkl"

if __name__ == '__main__':
    b=[]
    feature_f=open(feature_data,'rb')
    a=pickle.load(feature_f)
    for i in range(len(a)):
        each_data=a[i]
        b.append({
            "function1":"",
            "function2":" ".join([x[0] for x in list(each_data["code_features"].items())+list(each_data["context_features"].items())+list(each_data["pattern-features"].items())]),
            "target":each_data["target"],
            "function_id":i
        })
    sequence_f=open(sequence_data,'wb')
    pickle.dump(b,sequence_f)
    feature_f.close()
    sequence_f.close()
