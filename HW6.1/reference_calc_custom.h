#ifndef REFERENCE_H__
#define REFERENCE_H__

void reference_calc_custom(const uchar4* const h_sourceImg,
                    const size_t numRowsSource, const size_t numColsSource,
                    const uchar4* const h_destImg,
                      uchar4* const h_blendedImg,
					  const unsigned char* h_mask,
					  const unsigned char* h_border,
					  const unsigned char* h_interior);

#endif
