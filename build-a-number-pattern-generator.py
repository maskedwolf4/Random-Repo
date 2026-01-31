def number_pattern(n):

    if not isinstance(n, int):
        return 'Argument must be an integer value.'

    if n < 1:
        return 'Argument must be an integer greater than 0.'

    pattern = ''
    for i in range(n):
        i = i + 1
        if i == 1:
            pattern = pattern + str(i)  
        else:
            pattern = pattern + ' ' + str(i)  
    
    return pattern
        
print(number_pattern(4))


# output : 1 2 3 4


