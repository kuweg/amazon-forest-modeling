def test_forward_not_fall(test_model, fake_image):

    _ = test_model(fake_image)


def test_output_has_right_shape(test_model, conf, fake_image):

    expected_shape = (1, conf.num_classes)
    forward_shape = test_model(fake_image).shape

    assert forward_shape == expected_shape
