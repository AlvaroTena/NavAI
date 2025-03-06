import io
from collections import Counter

import rlnav.networks.ppo_networks as ppo_networks
import tensorflow as tf
from neptune import Run
from rlnav.agent.multiobjective_masked_ppo_agent import MultiObjectiveMaskedPPOAgent
from tf_agents.environments import TFPyEnvironment
from tf_agents.networks import layer_utils


def create_ppo_agent(
    train_env: TFPyEnvironment, npt_run: Run, rnn=True, learning_rate=1e-3
):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0, dtype=tf.int64)

    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    time_step_spec = train_env.time_step_spec()

    if rnn:
        actor_net = ppo_networks.create_actor_rnn_net(observation_spec, action_spec)
        value_net = ppo_networks.create_value_rnn_net(observation_spec)
    else:
        actor_net = ppo_networks.create_actor_net(observation_spec, action_spec)
        value_net = ppo_networks.create_value_net(observation_spec)

    agent = MultiObjectiveMaskedPPOAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=10,
        train_step_counter=train_step_counter,
        # entropy_regularization=0.1,
        normalize_rewards=True,
        use_gae=True,
        use_td_lambda_return=True,
        # debug_summaries=True,
        # summarize_grads_and_vars=True,
    )

    agent.initialize()

    def unify_repeated_cat_proj(model_summary: str) -> str:
        lines = model_summary.splitlines()
        cat_proj_blocks = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.strip().startswith(
                "CategoricalProjectionNetwork (CategoricalProjectionNetwork)"
            ):
                block_lines = [line]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].rstrip("\n")
                    if next_line.strip().startswith(
                        "|"
                    ) or next_line.strip().startswith("¯"):
                        block_lines.append(next_line)
                        j += 1
                    else:
                        break

                block_text = "\n".join(block_lines)
                cat_proj_blocks.append((i, j - 1, block_text))

                i = j
            else:
                i += 1

        block_counts = Counter(b[2] for b in cat_proj_blocks)

        placeholders = []
        i = 0
        block_index = 0
        while i < len(lines):
            if (
                block_index < len(cat_proj_blocks)
                and i == cat_proj_blocks[block_index][0]
            ):
                start_idx, end_idx, block_text = cat_proj_blocks[block_index]
                block_index += 1

                placeholders.append(("CATBLOCK", block_text))

                i = end_idx + 1
            else:
                placeholders.append(("LINE", lines[i]))
                i += 1

        output_lines = []
        seen_counts = Counter()
        for tk_type, tk_value in placeholders:
            if tk_type == "LINE":
                output_lines.append(tk_value)
            else:
                block_text = tk_value
                total_appearances = block_counts[block_text]

                seen_counts[block_text] += 1
                current_count = seen_counts[block_text]

                if total_appearances == 1:
                    output_lines.extend(block_text.split("\n"))
                    continue

                if current_count == 1:
                    output_lines.extend(block_text.split("\n"))

                if current_count == total_appearances:
                    output_lines.append(f"  ... (repeated {total_appearances} times)")

        return "\n".join(output_lines)

    summary = {}
    with io.StringIO() as s:
        layer_utils.print_summary(
            actor_net,
            line_length=150,
            print_fn=lambda x, **kwargs: s.write(x + "\n"),
            expand_nested=True,
        )
        model_summary = s.getvalue()
    summary["agent/actor_model"] = unify_repeated_cat_proj(model_summary)

    with io.StringIO() as s:
        layer_utils.print_summary(
            value_net,
            line_length=150,
            print_fn=lambda x, **kwargs: s.write(x + "\n"),
            expand_nested=True,
        )
        model_summary = s.getvalue()
    summary["agent/value_model"] = model_summary

    npt_run["training"] = summary

    return agent
