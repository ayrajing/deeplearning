#最长连续子序列及长度
def find_length_of_LCIS(nums):
    num_set=set(nums)
    max_length=0
    longest_sequence=[]
    for num in nums:
        if num-1 not in num_set:
            current_num=num
            current_length=1
            current_sequence=[current_num]
            while current_num+1 in num_set:
                current_num+=1
                current_length+=1
                current_sequence.append(current_num)
            if current_length>max_length:
                max_length=current_length
                longest_sequence=current_sequence
    return longest_sequence,max_length
            
