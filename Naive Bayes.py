data = []
test = 'it was a good game'

str_1 = 'a great game'
str_2 = 'the election was over'
str_3 = 'very clean match'
str_4 = 'a clean but forgettable game'
str_5 = 'it was a close election'
labels = [1.0,0.0,1.0,1.0,0.0]

data.append(str_1)
data.append(str_2)
data.append(str_3)
data.append(str_4)
data.append(str_5)

for ix in range(len(data)):
    data[ix] = data[ix].split(' ')


def get_probability(data):
    total_count = 0
    data_dict_yes = {}
    data_dict_no = {}
    for ix in range(len(data)):
        for word in data[ix]:
            if labels[ix] == 0:
                if word in data_dict_no:
                    data_dict_no[word] += 1.0
                else :
                    data_dict_no[word] = 1.0
                    if word in data_dict_yes:
                        pass
                    else:
                        total_count += 1.0
            else :
                if word in data_dict_yes:
                    data_dict_yes[word] += 1.0
                else :
                    data_dict_yes[word] = 1.0
                    if word in data_dict_no:
                        pass
                    else:
                        total_count += 1.0

    for ix in data_dict_yes:
        data_dict_yes[ix] /= total_count

    for ix in data_dict_no:
        data_dict_no[ix] /= total_count

    return data_dict_yes, data_dict_no


def get_prior(labels):
    p_sports = 0
    for ix in labels:
        p_sports += ix
    p_sports /= len(labels)
    p_not = 1 - p_sports
    return p_sports, p_not


def predict_func(test,data,labels):
    test = test.split()

    data_dict_yes, data_dict_no = get_probability(data)
    p_sports, p_not = get_prior(labels)
    test_0 = 1
    test_1 = 1

    for word in test:
        if word in data_dict_no.keys():
            test_0 *= data_dict_no[word]
        else :
            pass

        if word in data_dict_yes.keys():
            test_1 *= data_dict_yes[word]
        else :
            pass

    test_0 *= p_not
    test_1 *= p_sports

    return test_0,test_1

test_not_sports,test_sports = predict_func(test, data, labels)

print (test_not_sports, test_sports)
