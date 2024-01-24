"""
Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39#issuecomment-368239697.
"""


def f(output_size, ksize, stride):
    return (output_size - 1) * stride + ksize


def pix_gan_receptive_field():
    last_layer = f(output_size=1, ksize=4, stride=1)
    # Receptive field: 4
    fourth_layer = f(output_size=last_layer, ksize=4, stride=1)
    # Receptive field: 7
    third_layer = f(output_size=fourth_layer, ksize=4, stride=2)
    # Receptive field: 16
    second_layer = f(output_size=third_layer, ksize=4, stride=2)
    # Receptive field: 34
    first_layer = f(output_size=second_layer, ksize=4, stride=2)
    # Receptive field: 70
    return first_layer


def feat_gan_receptive_field():
    last_layer = f(output_size=1, ksize=4, stride=1)
    # Receptive field: 4
    third_layer = f(output_size=last_layer, ksize=4, stride=2)
    # Receptive field: 10
    second_layer = f(output_size=third_layer, ksize=4, stride=2)
    # Receptive field: 22
    first_layer = f(output_size=second_layer, ksize=4, stride=2)
    # Receptive field: 46

    # Including the segmentation model that creates the input feature maps:
    seg_last_layer = f(output_size=first_layer, ksize=2, stride=2)
    # Receptive field: 92
    seg_second_layer = f(output_size=seg_last_layer, ksize=3, stride=1)
    # Receptive field: 94
    seg_first_layer = f(output_size=seg_second_layer, ksize=3, stride=1)
    # Receptive field: 96

    return first_layer, seg_first_layer


if __name__ == "__main__":
    print("Pixel-level PatchGAN:", pix_gan_receptive_field())
    print(
        "Feature-level PatchGAN (excluding/including segmentation model):",
        feat_gan_receptive_field(),
    )
