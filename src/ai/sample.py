from random import Random

from puyopuyo.game import Field, PuyoPairGenerator, field_to_pretty_str
from puyopuyo.search import simulate_all_put_actions

puyo_pair_generator = PuyoPairGenerator(114514)
puyo_pairs = [puyo_pair_generator.generate_puyo_pair() for _ in range(30)]

rand = Random(1919)

field = Field()
max_chain = 0
score = 0
for seen_puyo_pairs in [puyo_pairs[i:i+2] for i in range(len(puyo_pairs))]:
    # actions = list_put_actions(field)
    # action = rand.choice(actions)
    results = simulate_all_put_actions(field, seen_puyo_pairs)
    best_result = max(results, key=lambda result: result.score)
    action = best_result.put_actions[0]

    field.put_puyo_pair(seen_puyo_pairs[0], action)
    chain_result = field.play_chains()

    score += chain_result.score
    max_chain = max(max_chain, chain_result.chains)

    if field.is_game_over():
        break

    print(field_to_pretty_str(field))
    print()

print('score: ', score)
print('max_chain: ', max_chain)