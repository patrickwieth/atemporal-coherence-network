from network import scheme_crafting

test = scheme_crafting.scheme_dice()

rollit = test.roll_dice()
print("behavior:", rollit)

result = test.create_scheme(rollit)
print(result)