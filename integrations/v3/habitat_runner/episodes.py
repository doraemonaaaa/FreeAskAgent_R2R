"""Episode selection and config loading."""

from pathlib import Path


def parse_episode_ids(spec):
    """``"1,2,3"`` or ``"@file"`` -> list of id strings, in order."""
    if not spec:
        return None
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text().splitlines()
        items = [line.split("#", 1)[0].strip() for line in lines]
        items = [item.split(",", 1)[0].strip() for item in items if item]
    else:
        items = [item.strip() for item in spec.split(",") if item.strip()]
    return items or None


def select_episodes(
    available,
    *,
    episode_id,
    episode_count,
    rank,
    world_size,
    episode_ids=None,
):
    """Select one exact episode, an explicit id list, or the eval prefix.

    ``episode_ids`` keeps the list's own order so a stratified evaluation
    set is sharded evenly across ranks category by category.
    """
    available = list(available)
    if episode_ids:
        by_id = {str(episode.episode_id): episode for episode in available}
        missing = [item for item in episode_ids if item not in by_id]
        if missing:
            raise ValueError(
                "episode ids not present in this split/scene: {}".format(
                    ", ".join(missing)
                )
            )
        selected = [by_id[item] for item in episode_ids]
    elif episode_id is not None:
        selected = [
            episode
            for episode in available
            if str(episode.episode_id) == str(episode_id)
        ]
        if not selected:
            raise ValueError(
                "episode_id {} is not present in this split/scene".format(
                    episode_id
                )
            )
    else:
        selected = (
            available
            if episode_count == 0
            else available[:episode_count]
        )
    return selected[rank::world_size]


def load_config(path, rank=0):
    """YAML -> (argparse defaults, agent environment). CLI and preset env win.

    Delegates to integrations/v3/run_config.py, the single parser of
    config.yaml shared with the launcher and serving scripts. ``rank`` selects
    the replica when a service lists several ``base_urls``.
    """
    from integrations.v3.run_config import RunConfig

    config = RunConfig.load(path)
    return config.argparse_defaults(rank), config.agent_env(rank)

