from collections import Counter

def show_results(counter):
    for digit in counter:
        print('%s : %d' % (digit, counter[digit]))

# get the sum of the number
def get_sum(n):
    # empty number
    sum = 0
    # take one item from the list
    for digit in str(n): 
        # add this number to the sum
        sum += int(digit)      
    return sum

# get the number of group with id = 0
def group_number(n_customers):
    customers = []
    # create a list with consumer ids
    for i in range(n_customers):
        customers.append(i)
    # list of all groups that customers will fall into
    groups_list = []
    # convert ids to groups
    for customer in customers:
        # get sum of the customer_id
        number_id = get_sum(customer)
        # add the group to the list of all group
        groups_list.append(number_id)
    # count how many people are in each group
    return(Counter(groups_list))

# get the number of group with the selected id
def group_number_w_fId(n_customers, n_first_id):
    customers = []
    # find last number for the function range
    n_last_id = n_first_id + n_customers
    # create a list with consumer ids
    for i in range(n_first_id, n_last_id):
        customers.append(i)
    # list of all groups that customers will fall into
    groups_list = []
    # convert ids to groups
    for customer in customers:
        # get sum of the customer_id
        number_id = get_sum(customer)
        # add the group to the list of all group
        groups_list.append(number_id)
        # print(groups_list)
    # count how many people are in each group
    return(Counter(groups_list))

groups_w_0 = group_number(20)
groups_w_id = group_number_w_fId(678, 325123)

show_results(groups_w_0)
print("************")
show_results(groups_w_id)

